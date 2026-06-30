from specula.base_processing_obj import OutputDesc
from specula.data_objects.subap_data import SubapData
from specula.lib.utils import unravel_index_2d
from specula.processing_objects.slopec import Slopec


class FsmHybridSlopec(Slopec):
    """
    FSM-Guided Kinematic Hybrid Tracker processing object.
    Implements a Dual-Brain architecture (Radar EMA + Instantaneous Sniper)
    with an Asymmetric Kinematic Leash to compute Shack-Hartmann slopes robustly 
    in extreme low-SNR, flickering, and high-dynamics closed-loop environments.
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

        # 1. Update Exponential Moving Averages (EMA)
        # Flush the EMA memory for subapertures that are starting completely fresh (Hard Drop)
        reset_ema = (self.miss_counter >= self.max_missed_frames)[:, None, None]

        self.ema_pixels = self.xp.where(
            reset_ema, pixels,
            self.ema_alpha * pixels + (1.0 - self.ema_alpha) * self.ema_pixels
        )

        # 2. Base Spatial Correlation (The Likelihood) via FFT
        fft_pixels = self.xp.fft.fft2(pixels, axes=(1, 2))
        corr_map = self.xp.fft.ifft2(fft_pixels * self.xp.conj(self.fft_template), axes=(1, 2)).real

        self.ema_corr = self.xp.where(
            reset_ema, corr_map, 
            self.ema_alpha * corr_map + (1.0 - self.ema_alpha) * self.ema_corr
        )

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

        x_pred = self.state_x1 + 0.5 * v_global_x
        y_pred = self.state_y1 + 0.5 * v_global_y

        # ------------------------------------------------------------------
        # THE SNIPER (Instantaneous Single Frame Detection)
        # ------------------------------------------------------------------
        c_mean_single = self.xp.mean(corr_map, axis=(1, 2))
        c_std_single = self.xp.maximum(self.xp.std(corr_map, axis=(1, 2)), 1e-6)

        # Bypass prior if Radar confirms an extremely strong signal
        override_mask = (snr_radar > self.snr_strong_thr) | ~self.is_locked
        prior_gauss = self._generate_gaussian(x_pred - self.offset, y_pred - self.offset, self.prior_sigma)
        spatial_prior = self.xp.where(
            override_mask[:, None, None],
            self.xp.ones_like(prior_gauss),
            (1.0 - self.prior_floor) * prior_gauss + self.prior_floor
        )

        weighted_corr = corr_map * spatial_prior
        flat_idx_single = self.xp.argmax(weighted_corr.reshape(n_subaps, -1), axis=1)

        c_max_single = corr_map.reshape(n_subaps, -1)[self.xp.arange(n_subaps), flat_idx_single]
        snr_sniper = (c_max_single - c_mean_single) / c_std_single
        sniper_yes = snr_sniper >= self.snr_thr

        y_idx_single = flat_idx_single // np_sub
        x_idx_single = flat_idx_single % np_sub
        x_coarse_single = x_idx_single + self.offset
        y_coarse_single = y_idx_single + self.offset

        # ------------------------------------------------------------------
        # THREE-TRACK CONFIDENCE LOGIC
        # ------------------------------------------------------------------
        use_track2 = ~sniper_yes & radar_yes
        use_track3 = ~sniper_yes & ~radar_yes
        
        # CRITICAL FIX: A valid hit is ANY hit with sufficient SNR. 
        # The kinematic leash restrains the WCoG mask, but DOES NOT invalidate the hit.
        valid_hit = ~use_track3

        # Choose baseline target source coordinates from the corresponding track
        x_coarse_raw = self.xp.where(use_track2, x_coarse_ema, x_coarse_single)
        y_coarse_raw = self.xp.where(use_track2, y_coarse_ema, y_coarse_single)

        # ------------------------------------------------------------------
        # ASYMMETRIC SPRING LEASH CONSTRAINT
        # Allows proportional speed towards the center, clips outwards noise.
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
        # FINE TRACKING (Dynamic WCoG)
        # ------------------------------------------------------------------
        # Wcog data source changes based on track confidence to prevent noise division
        wcog_pixels = self.xp.where(use_track2[:, None, None], self.ema_pixels, pixels)

        dynamic_weight = self._generate_gaussian(x_coarse, y_coarse, self.fwhm_pix)
        weighted_img = self.xp.maximum(wcog_pixels, 0.0) * dynamic_weight
        flux_sum = self.xp.sum(weighted_img, axis=(1, 2))

        x_est = self.xp.sum(self.xx[None, :, :] * weighted_img, axis=(1, 2)) / self.xp.maximum(flux_sum, 1e-6)
        y_est = self.xp.sum(self.yy[None, :, :] * weighted_img, axis=(1, 2)) / self.xp.maximum(flux_sum, 1e-6)

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
        valid_trk = trk_mask & valid_hit
        invalid_trk = trk_mask & ~valid_hit

        new_miss_counter[valid_trk] = 0
        new_state_x1[valid_trk] = x_est[valid_trk]
        new_state_y1[valid_trk] = y_est[valid_trk]

        # Soft-Hold (Track 3)
        new_miss_counter[invalid_trk] += 1
        new_state_x1[invalid_trk] = x_pred[invalid_trk]
        new_state_y1[invalid_trk] = y_pred[invalid_trk]

        # --- C. DROP LOGIC ---
        lock_lost = invalid_trk & (new_miss_counter >= self.max_missed_frames)
        new_is_locked[lock_lost] = False
        new_lock_counter[lock_lost] = 0
        
        # Ensure miss_counter is zeroed out for all unlocked subapertures
        new_miss_counter[~new_is_locked] = 0

        # --- Update Global Memory ---
        self.state_x2 = self.state_x1.copy()
        self.state_y2 = self.state_y1.copy()
        self.state_x1 = new_state_x1
        self.state_y1 = new_state_y1
        self.is_locked = new_is_locked
        self.lock_counter = new_lock_counter
        self.miss_counter = new_miss_counter

        # --- Output Slopes to the Reconstructor ---
        norm_factor = np_sub / 2.0

        raw_slope_x = (self.state_x1 - cntrd) / norm_factor
        raw_slope_y = (self.state_y1 - cntrd) / norm_factor

        # Output exactly 0.0 if not locked to maintain DM inertia
        self.slopes.xslopes = self.xp.where(self.is_locked, raw_slope_x, 0.0)
        self.slopes.yslopes = self.xp.where(self.is_locked, raw_slope_y, 0.0)
        self.slopes.generation_time = self.current_time

        # Update telemetry
        self.flux_per_subaperture_vector.value[:] = flux_sum
        self.total_counts.value[0] = self.xp.sum(flux_sum)
        self.subap_counts.value[0] = self.xp.mean(flux_sum)

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
