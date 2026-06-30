from specula.base_processing_obj import OutputDesc
from specula.data_objects.subap_data import SubapData
from specula.lib.utils import unravel_index_2d
from specula.processing_objects.slopec import Slopec


class FsmHybridSlopec(Slopec):
    """
    FSM-Guided Kinematic Hybrid Tracker processing object.

    Implements a Dual-Brain architecture that balances an instantaneous "Sniper" 
    target extractor against a historical Exponential Moving Average (EMA) "Radar".
    Designed specifically to guarantee robust Shack-Hartmann centroiding and closed-loop 
    stability in extreme low-SNR target environments subject to photon Poisson 
    flickering and aggressive mirror dynamics.

    Parameters
    ----------
    subapdata : SubapData
        SPECULA object containing the sub-aperture geometry, pixel indexing, 
        and valid pupil masks.
    fwhm_pix : float [pixels], optional
        Full Width at Half Maximum of the expected nominal spot profile, 
        expressed in pixels. Used to scale the matched filter template and 
        the dynamic WCoG gaussian weighting window. Default is 1.5.
    snr_thr : float [1], optional
        Significance threshold for the signal-to-noise ratio. Applies to both 
        the instantaneous sniper map and the historical radar map to declare 
        a valid detection. Default is 3.5.
    snr_strong_thr : float [1], optional
        SNR threshold above which a signal is considered absolutely dominant. 
        When exceeded, the spatial prior kinematic mask is bypassed to prevent 
        spatial lag during massive, unpredicted slews. Default is 15.0.
    prior_sigma : float [pixels], optional
        Standard deviation (in pixels) of the Gaussian spatial prior mask. 
        Defines the statistical search radius around the kinematically 
        predicted spot center during tracking mode. Default is 5.0.
    prior_floor : float [1], optional
        The baseline probability transmission floor of the spatial prior mask. 
        Guarantees that a valid spot far from the prediction is attenuated 
        but never mathematically zeroed out. Default is 0.10.
    lock_frames_req : int [1], optional
        Hysteresis parameter defining the number of consecutive, spatially 
        consistent valid frames required to transition the Finite State Machine 
        from Acquisition to Tracking mode. Default is 3.
    max_missed_frames : int [1], optional
        The maximum number of consecutive invalid frames allowed before the FSM 
        declares a total fading state, drops the lock, and flushes the 
        historical memory. Default is 10.
    max_v : float [pixels/frame], optional
        Maximum physical slew rate expected per frame under nominal atmospheric 
        and mechanical perturbations, in pixels. Used as the baseline bounding 
        limit for the kinematic leash and global tip-tilt prediction. Default is 0.5.
    leash_alpha : float [1], optional
        Proportional gain of the asymmetric closed-loop spring leash. Scales the 
        permissible spot step size towards the nominal center of the sub-aperture, 
        modeling the corrector mirror's restorative force. Default is 0.5.
    acq_radius_sq : float [pixels^2], optional
        Maximum squared spatial distance (in square pixels) permitted between 
        consecutive centroid estimates during acquisition, or between the sniper 
        and radar peaks, to declare spatial consistency. Default is 4.0.
    ema_alpha : float [1], optional
        The smoothing factor of the Exponential Moving Average filter. Determines 
        the temporal weight of the current frame relative to history for the 
        background "Radar" image accumulation. Default is 0.3.
    **kwargs : dict
        Additional keyword arguments passed to the base `Slopec` constructor 
        (e.g., `target_device_idx`, `dtype`).
    """

    def __init__(self,
                 subapdata: SubapData,
                 fwhm_pix: float = 1.5,
                 snr_thr: float = 3.5,
                 snr_strong_thr: float = 15.0,
                 prior_sigma: float = 5.0,
                 prior_floor: float = 0.10,
                 lock_frames_req: int = 3,
                 max_missed_frames: int = 10,
                 max_v: float = 0.5,
                 leash_alpha: float = 0.5,
                 acq_radius_sq: float = 4.0,
                 ema_alpha: float = 0.3,
                 **kwargs):

        self.subapdata = subapdata
        super().__init__(**kwargs)

        # Tracker physical parameters
        self.fwhm_pix = fwhm_pix
        self.snr_thr = snr_thr
        self.snr_strong_thr = snr_strong_thr
        self.prior_sigma = prior_sigma
        self.prior_floor = prior_floor

        # Hysteresis, kinematic limits, and Temporal memory
        self.lock_frames_req = lock_frames_req
        self.max_missed_frames = max_missed_frames
        self.max_v = max_v
        self.leash_alpha = leash_alpha
        self.acq_radius_sq = acq_radius_sq
        self.ema_alpha = ema_alpha

        # Output declarations
        self.outputs['out_subapdata'] = self.subapdata
        self.slopes.single_mask = self.subapdata.single_mask()
        self.slopes.display_map = self.subapdata.display_map

        # --- Memory Allocation for FSM, Kinematics, and Moving Average ---
        n_subaps = self.nsubaps()
        np_sub = self.subapdata.np_sub
        cntrd = (np_sub - 1) / 2.0

        # State vectors (absolute pixel coordinates)
        self.state_x1 = self.xp.full(n_subaps, cntrd, dtype=self.dtype)
        self.state_x2 = self.xp.full(n_subaps, cntrd, dtype=self.dtype)
        self.state_y1 = self.xp.full(n_subaps, cntrd, dtype=self.dtype)
        self.state_y2 = self.xp.full(n_subaps, cntrd, dtype=self.dtype)

        # FSM counters
        self.lock_counter = self.xp.zeros(n_subaps, dtype=self.xp.int32)
        self.miss_counter = self.xp.zeros(n_subaps, dtype=self.xp.int32)
        self.is_locked = self.xp.zeros(n_subaps, dtype=self.xp.bool_)

        # Exponential Moving Average buffers (The "Radar")
        self.ema_pixels = self.xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)
        self.ema_corr = self.xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)

        # --- Grids Generation ---
        self.x_grid = self.xp.arange(np_sub, dtype=self.dtype)
        self.y_grid = self.xp.arange(np_sub, dtype=self.dtype)
        self.xx, self.yy = self.xp.meshgrid(self.x_grid, self.y_grid)

        # --- Static Analytical Template (FFT Centered & Shifted) ---
        half_np = np_sub // 2
        dx_wrap = self.xp.where(self.x_grid > half_np - 1, self.x_grid - np_sub, self.x_grid)
        dy_wrap = self.xp.where(self.y_grid > half_np - 1, self.y_grid - np_sub, self.y_grid)
        xx_wrap, yy_wrap = self.xp.meshgrid(dx_wrap, dy_wrap)

        self.offset = 0.5 if np_sub % 2 == 0 else 0.0
        sigma = self.fwhm_pix / (2.0 * self.xp.sqrt(2.0 * self.xp.log(2.0)))
        template_centered = self.xp.exp(-((xx_wrap - self.offset)**2 + (yy_wrap - self.offset)**2) / (2 * sigma**2))
        template_centered /= self.xp.sum(template_centered)

        self.fft_template = self.xp.fft.fft2(template_centered[None, :, :], axes=(1, 2))

    @classmethod
    def output_names(cls):
        result = super().output_names()
        result.update({
            'out_subapdata': OutputDesc(SubapData, 'Subaperture data with geometry information')
        })
        return result

    def nsubaps(self):
        return self.subapdata.n_subaps

    def nslopes(self):
        return self.subapdata.n_subaps * 2

    @property
    def subap_idx(self):
        return self.subapdata.idxs

    def _generate_gaussian(self, x_pos, y_pos, fwhm):
        sigma = fwhm / (2.0 * self.xp.sqrt(2.0 * self.xp.log(2.0)))
        if self.xp.isscalar(x_pos):
            dx = self.xx - x_pos
            dy = self.yy - y_pos
        else:
            dx = self.xx[None, :, :] - x_pos[:, None, None]
            dy = self.yy[None, :, :] - y_pos[:, None, None]
        return self.xp.exp(-(dx**2 + dy**2) / (2 * sigma**2))

    def trigger_code(self):
        self.calc_slopes_nofor()

    def calc_slopes_nofor(self):
        if self.subapdata is None:
            self.logger.warning('subapdata is not valid.')
            return

        in_pixels = self.local_inputs['in_pixels'].pixels
        n_subaps = self.nsubaps()
        np_sub = self.subapdata.np_sub
        cntrd = (np_sub - 1) / 2.0

        # 0. Extract raw pixels
        idx2d = unravel_index_2d(self.subap_idx, in_pixels.shape, self.xp)
        pixels = in_pixels[idx2d].reshape(n_subaps, np_sub, np_sub).astype(self.dtype)

        raw_flux_sum = self.xp.sum(pixels, axis=(1, 2))

        # 1. Update Exponential Moving Averages (EMA)
        self.ema_pixels = self.ema_alpha * pixels + (1.0 - self.ema_alpha) * self.ema_pixels

        # 2. Base Spatial Correlation (The Likelihood) via FFT
        fft_pixels = self.xp.fft.fft2(pixels, axes=(1, 2))
        corr_map = self.xp.fft.ifft2(fft_pixels * self.xp.conj(self.fft_template), axes=(1, 2)).real

        self.ema_corr = self.ema_alpha * corr_map + (1.0 - self.ema_alpha) * self.ema_corr

        # ------------------------------------------------------------------
        # THE RADAR (EMA Detection)
        # Evaluates the global presence of the star over recent frames.
        # ------------------------------------------------------------------
        ema_mean = self.xp.mean(self.ema_corr, axis=(1, 2))
        ema_std = self.xp.maximum(self.xp.std(self.ema_corr, axis=(1, 2)), 1e-6)

        flat_idx_ema = self.xp.argmax(self.ema_corr.reshape(n_subaps, -1), axis=1)
        c_max_ema = self.ema_corr.reshape(n_subaps, -1)[self.xp.arange(n_subaps), flat_idx_ema]

        snr_radar = (c_max_ema - ema_mean) / ema_std
        radar_yes = snr_radar >= self.snr_thr

        y_idx_ema = flat_idx_ema // np_sub
        x_idx_ema = flat_idx_ema % np_sub
        x_coarse_ema = x_idx_ema + self.offset
        y_coarse_ema = y_idx_ema + self.offset

        single_mean = self.xp.mean(corr_map, axis=(1, 2))
        single_std = self.xp.maximum(self.xp.std(corr_map, axis=(1, 2)), 1e-6)

        flat_idx_raw = self.xp.argmax(corr_map.reshape(n_subaps, -1), axis=1)
        c_max_raw = corr_map.reshape(n_subaps, -1)[self.xp.arange(n_subaps), flat_idx_raw]
        snr_sniper_unweighted = (c_max_raw - single_mean) / single_std

        # ------------------------------------------------------------------
        # KINEMATIC PREDICTION (Swarm Tracking)
        # ------------------------------------------------------------------
        if self.xp.any(self.is_locked):
            vx_valid = self.state_x1[self.is_locked] - self.state_x2[self.is_locked]
            vy_valid = self.state_y1[self.is_locked] - self.state_y2[self.is_locked]
            v_global_x = self.xp.clip(self.xp.median(vx_valid), -self.max_v, self.max_v)
            v_global_y = self.xp.clip(self.xp.median(vy_valid), -self.max_v, self.max_v)
        else:
            v_global_x, v_global_y = 0.0, 0.0

        x_pred = self.state_x1 + v_global_x
        y_pred = self.state_y1 + v_global_y

        # ------------------------------------------------------------------
        # BAYESIAN PRIOR & WEIGHTED SNIPER
        # ------------------------------------------------------------------
        override_mask = (snr_radar > self.snr_strong_thr) | (snr_sniper_unweighted > self.snr_strong_thr) | ~self.is_locked
        prior_gauss = self._generate_gaussian(x_pred, y_pred, self.prior_sigma)
        spatial_prior = self.xp.where(
            override_mask[:, None, None],
            self.xp.ones_like(prior_gauss),
            (1.0 - self.prior_floor) * prior_gauss + self.prior_floor
        )

        weighted_corr = corr_map * spatial_prior
        flat_idx_single = self.xp.argmax(weighted_corr.reshape(n_subaps, -1), axis=1)

        c_max_single = corr_map.reshape(n_subaps, -1)[self.xp.arange(n_subaps), flat_idx_single]
        snr_sniper = (c_max_single - single_mean) / single_std
        sniper_yes = snr_sniper >= self.snr_thr

        y_idx_single = flat_idx_single // np_sub
        x_idx_single = flat_idx_single % np_sub
        x_coarse_single = x_idx_single + self.offset
        y_coarse_single = y_idx_single + self.offset

        # ------------------------------------------------------------------
        # THREE-TRACK LOGIC & SPRING LEASH
        # ------------------------------------------------------------------
        dist_single_to_ema_sq = (x_coarse_single - x_coarse_ema)**2 + (y_coarse_single - y_coarse_ema)**2
        sniper_consistent = dist_single_to_ema_sq <= self.acq_radius_sq

        # Track 1: Sniper YES and is either already tracking or consistent with Radar
        use_track1 = sniper_yes & (self.is_locked | sniper_consistent)

        # Track 2: Fallback to Radar if Sniper fails or is inconsistent
        use_track2 = radar_yes & ~use_track1

        # Track 3: Total signal loss
        use_track3 = ~radar_yes & ~use_track1

        # Choose baseline target source coordinates from the corresponding track
        x_coarse_raw = self.xp.where(use_track2, x_coarse_ema, x_coarse_single)
        y_coarse_raw = self.xp.where(use_track2, y_coarse_ema, y_coarse_single)

        # ------------------------------------------------------------------
        # ASYMMETRIC SPRING LEASH CONSTRAINT
        # Allows proportional speed towards the center, clips outwards noise.
        # This constraint ONLY shapes the WCoG mask position; it does NOT invalidate the hit.
        # ------------------------------------------------------------------
        dx_raw = x_coarse_raw - self.state_x1
        dy_raw = y_coarse_raw - self.state_y1

        x_rel = self.state_x1 - cntrd
        y_rel = self.state_y1 - cntrd

        # X-Axis limits
        limit_pos_x = self.xp.where(x_rel > 0, self.max_v, self.xp.maximum(self.max_v, self.leash_alpha * self.xp.abs(x_rel)))
        limit_neg_x = self.xp.where(x_rel > 0, -self.xp.maximum(self.max_v, self.leash_alpha * self.xp.abs(x_rel)), -self.max_v)

        # Y-Axis limits
        limit_pos_y = self.xp.where(y_rel > 0, self.max_v, self.xp.maximum(self.max_v, self.leash_alpha * self.xp.abs(y_rel)))
        limit_neg_y = self.xp.where(y_rel > 0, -self.xp.maximum(self.max_v, self.leash_alpha * self.xp.abs(y_rel)), -self.max_v)

        dx_clipped = self.xp.clip(dx_raw, limit_neg_x, limit_pos_x)
        dy_clipped = self.xp.clip(dy_raw, limit_neg_y, limit_pos_y)

        x_coarse = self.xp.where(self.is_locked, self.state_x1 + dx_clipped, x_coarse_raw)
        y_coarse = self.xp.where(self.is_locked, self.state_y1 + dy_clipped, y_coarse_raw)

        # ------------------------------------------------------------------
        # FINE TRACKING (Iterative Dynamic WCoG)
        # ------------------------------------------------------------------
        wcog_pixels = self.xp.where(use_track2[:, None, None], self.ema_pixels, pixels)

        # Pass 1: Coarse centering (Subject to Grid Bias)
        dynamic_weight_1 = self._generate_gaussian(x_coarse, y_coarse, self.fwhm_pix)
        weighted_img_1 = self.xp.maximum(wcog_pixels, 0.0) * dynamic_weight_1

        flux_sum_1 = self.xp.sum(weighted_img_1, axis=(1, 2))
        x_est_1 = self.xp.sum(self.xx[None, :, :] * weighted_img_1, axis=(1, 2)) / self.xp.maximum(flux_sum_1, 1e-6)
        y_est_1 = self.xp.sum(self.yy[None, :, :] * weighted_img_1, axis=(1, 2)) / self.xp.maximum(flux_sum_1, 1e-6)

        zero_flux_mask = flux_sum_1 < 1e-6
        x_est_1 = self.xp.where(zero_flux_mask, x_coarse.astype(self.dtype), x_est_1)
        y_est_1 = self.xp.where(zero_flux_mask, y_coarse.astype(self.dtype), y_est_1)

        # Pass 2: Fine sub-pixel centering (Eliminates Grid Bias)
        dynamic_weight_2 = self._generate_gaussian(x_est_1, y_est_1, self.fwhm_pix)
        weighted_img_2 = self.xp.maximum(wcog_pixels, 0.0) * dynamic_weight_2

        flux_sum = self.xp.sum(weighted_img_2, axis=(1, 2))
        x_est = self.xp.sum(self.xx[None, :, :] * weighted_img_2, axis=(1, 2)) / self.xp.maximum(flux_sum, 1e-6)
        y_est = self.xp.sum(self.yy[None, :, :] * weighted_img_2, axis=(1, 2)) / self.xp.maximum(flux_sum, 1e-6)

        # Fallback in case of total numerical failure
        x_est = self.xp.where(zero_flux_mask, x_est_1, x_est)
        y_est = self.xp.where(zero_flux_mask, y_est_1, y_est)

        # Protection against math errors
        zero_flux_mask = flux_sum < 1e-6
        x_est = self.xp.where(zero_flux_mask, x_coarse.astype(self.dtype), x_est)
        y_est = self.xp.where(zero_flux_mask, y_coarse.astype(self.dtype), y_est)

        # ========================================================
        # FINITE STATE MACHINE (Vectorized Updates)
        # ========================================================
        new_is_locked = self.is_locked.copy()
        new_lock_counter = self.lock_counter.copy()
        new_miss_counter = self.miss_counter.copy()
        new_state_x1 = self.state_x1.copy()
        new_state_y1 = self.state_y1.copy()

        # --- A. ACQUISITION MODE ---
        acq_mask = ~self.is_locked
        valid_hit = ~use_track3
        valid_acq = acq_mask & valid_hit
        invalid_acq = acq_mask & ~valid_hit

        dist_sq = (x_est - self.state_x1)**2 + (y_est - self.state_y1)**2
        is_consistent = dist_sq <= self.acq_radius_sq
        is_first_hit = self.lock_counter == 0

        consistent_acq = valid_acq & (is_consistent | is_first_hit)
        inconsistent_acq = valid_acq & ~is_consistent

        new_lock_counter[consistent_acq] += 1
        new_lock_counter[inconsistent_acq] = 1
        new_lock_counter[invalid_acq] = 0

        new_state_x1[valid_acq] = x_est[valid_acq]
        new_state_y1[valid_acq] = y_est[valid_acq]

        lock_achieved = valid_acq & (new_lock_counter >= self.lock_frames_req)
        new_is_locked[lock_achieved] = True
        new_miss_counter[lock_achieved] = 0

        # --- B. TRACKING MODE ---
        trk_mask = self.is_locked

        # We consider a tracking frame fully nominal ONLY if the instantaneous Sniper sees it.
        # If we are relying on Track 2 (EMA fallback), we still update the state with x_est
        # (which comes from EMA pixels) to keep the loop moving, but we let the miss_counter rise!
        nominal_trk = trk_mask & use_track1
        flicker_trk = trk_mask & use_track2
        fading_trk  = trk_mask & use_track3

        # Nominal Tracking: Zero misses, update state with instant WCoG
        new_miss_counter[nominal_trk] = 0
        new_state_x1[nominal_trk] = x_est[nominal_trk]
        new_state_y1[nominal_trk] = y_est[nominal_trk]

        # Flicker Tracking (EMA Fallback): Increment miss, but STILL update state with EMA WCoG!
        # This keeps the slopes fluid and active, preventing DM steps while accounting for the miss.
        new_miss_counter[flicker_trk] += 1
        new_state_x1[flicker_trk] = x_est[flicker_trk]
        new_state_y1[flicker_trk] = y_est[flicker_trk]

        # Total Fading (Track 3): Increment miss, blind cinematic prediction update
        new_miss_counter[fading_trk] += 1
        new_state_x1[fading_trk] = x_pred[fading_trk]
        new_state_y1[fading_trk] = y_pred[fading_trk]

        # --- C. DROP LOGIC & EMA FLUSH ---
        invalid_trk = trk_mask & ~use_track1
        lock_lost = invalid_trk & (new_miss_counter >= self.max_missed_frames)
        new_is_locked[lock_lost] = False
        new_lock_counter[lock_lost] = 0

        # Flush ghost images exactly upon dropping lock
        ll_mask = lock_lost[:, None, None]
        self.ema_pixels = self.xp.where(ll_mask, pixels, self.ema_pixels)
        self.ema_corr = self.xp.where(ll_mask, corr_map, self.ema_corr)

        new_miss_counter[~new_is_locked] = 0

        # --- Update Global Memory ---
        self.state_x2 = self.state_x1.copy()
        self.state_y2 = self.state_y1.copy()
        self.state_x1 = new_state_x1
        self.state_y1 = new_state_y1
        self.is_locked = new_is_locked
        self.lock_counter = new_lock_counter
        self.miss_counter = new_miss_counter

        # --- Output Slopes (SAFEGUARD RESTORED) ---
        norm_factor = np_sub / 2.0

        raw_slope_x = (self.state_x1 - cntrd) / norm_factor
        raw_slope_y = (self.state_y1 - cntrd) / norm_factor

        # THIS PREVENTS DM EXPLOSION DURING ACQUISITION!
        self.slopes.xslopes = self.xp.where(self.is_locked, raw_slope_x, 0.0)
        self.slopes.yslopes = self.xp.where(self.is_locked, raw_slope_y, 0.0)
        self.slopes.generation_time = self.current_time

        # Update Telemetry
        self.flux_per_subaperture_vector.value[:] = raw_flux_sum
        self.total_counts.value[0] = self.xp.sum(raw_flux_sum)
        self.subap_counts.value[0] = self.xp.mean(raw_flux_sum)

        # plot for debugging
        debug_plot = False
        #if self.t_to_seconds(self.current_time) < 1.75:
        #    debug_plot = False
        #else:
        #    debug_plot = True
        if debug_plot:
            import matplotlib.pyplot as plt
            from specula import cpuArray
            subap_to_plot = 0

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            is_locked_curr = bool(self.is_locked[subap_to_plot])
            t2 = bool(use_track2[subap_to_plot])
            t3 = bool(use_track3[subap_to_plot])

            if t3:
                track_str = "Track 3: HOLD (Fading)"
            elif t2:
                track_str = "Track 2: EMA WCoG (Flickering)"
            else:
                track_str = "Track 1: INSTANT WCoG (Strong Hit)"

            # --- 1. Raw Pixel Image (Instant Sniper) ---
            ax = axes[0, 0]
            im1 = ax.imshow(cpuArray(pixels[subap_to_plot]), cmap='hot')
            ax.axhline(cntrd, color='w', linestyle=':', alpha=0.3)
            ax.axvline(cntrd, color='w', linestyle=':', alpha=0.3)

            ax.plot(cpuArray(x_est[subap_to_plot]),
                    cpuArray(y_est[subap_to_plot]), 'g*', markersize=16, label='Final Est')
            ax.set_title(f'Subap {subap_to_plot}: SINGLE Raw Pixels')
            plt.colorbar(im1, ax=ax)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            # --- 2. Single Correlation Map (Sniper) ---
            ax = axes[0, 1]
            im2 = ax.imshow(cpuArray(corr_map[subap_to_plot]), cmap='viridis')
            ax.axhline(cntrd, color='w', linestyle=':', alpha=0.3)
            ax.axvline(cntrd, color='w', linestyle=':', alpha=0.3)

            ax.plot(cpuArray(x_idx_single[subap_to_plot]), cpuArray(y_idx_single[subap_to_plot]),
                    marker='o', color='red', markerfacecolor='none',
                    markersize=16, markeredgewidth=1.0, linestyle='', label='Instant Peak')

            if is_locked_curr:
                ax.plot(cpuArray(x_pred[subap_to_plot]) - self.offset,
                        cpuArray(y_pred[subap_to_plot]) - self.offset,
                        marker='s', color='cyan', markerfacecolor='none', 
                        markersize=16, markeredgewidth=1.0, linestyle='',
                        label='Prediction')

            ax.set_title(f'SINGLE Correlation Map\nSniper SNR = {snr_sniper[subap_to_plot]:.2f}')
            plt.colorbar(im2, ax=ax)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            # --- 3. EMA Pixels (Radar Memory) ---
            ax = axes[1, 0]
            im3 = ax.imshow(cpuArray(self.ema_pixels[subap_to_plot]), cmap='hot')
            ax.axhline(cntrd, color='w', linestyle=':', alpha=0.3)
            ax.axvline(cntrd, color='w', linestyle=':', alpha=0.3)
            
            ax.plot(cpuArray(x_est[subap_to_plot]),
                    cpuArray(y_est[subap_to_plot]), 'g*', markersize=16, label='Final Est')
            ax.set_title(f'Subap {subap_to_plot}: EMA Pixels')
            plt.colorbar(im3, ax=ax)

            # --- 4. EMA Correlation Map (Radar) ---
            ax = axes[1, 1]
            im4 = ax.imshow(cpuArray(self.ema_corr[subap_to_plot]), cmap='viridis')
            ax.axhline(cntrd, color='w', linestyle=':', alpha=0.3)
            ax.axvline(cntrd, color='w', linestyle=':', alpha=0.3)
            
            ax.plot(cpuArray(x_idx_ema[subap_to_plot]),
                    cpuArray(y_idx_ema[subap_to_plot]),
                    marker='D', color='orange', markerfacecolor='none',
                    markersize=16, markeredgewidth=1.0, linestyle='', label='EMA Peak')
            
            ax.set_title(f'EMA Correlation Map\nRadar SNR = {snr_radar[subap_to_plot]:.2f}')
            plt.colorbar(im4, ax=ax)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            color_title = 'green' if is_locked_curr else 'red'
            fig.suptitle(f'FSM State: Locked={is_locked_curr} | L_Count={self.lock_counter[subap_to_plot]} | M_Count={self.miss_counter[subap_to_plot]}\nActive Logic: {track_str}',
                         fontsize=14, fontweight='bold', color=color_title)

            plt.tight_layout()
            plt.show()

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_subapdata'].generation_time = self.current_time
