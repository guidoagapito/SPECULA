from specula.base_processing_obj import OutputDesc
from specula.data_objects.subap_data import SubapData
from specula.lib.utils import unravel_index_2d
from specula.processing_objects.slopec import Slopec


class FsmHybridSlopec(Slopec):
    """
    FSM-Guided Kinematic Hybrid Tracker processing object.
    Computes Shack-Hartmann slopes using Fast Fourier Transform (FFT) Correlation,
    a global kinematic predictor (Swarm Tracking), and a Finite State Machine 
    for extreme low-SNR target environments.
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
                 fast_relock_frames: int = 3,
                 max_v: float = 0.5,
                 acq_radius_sq: float = 2.0,
                 **kwargs):

        self.subapdata = subapdata
        super().__init__(**kwargs)

        # Tracker physical parameters
        self.fwhm_pix = fwhm_pix
        self.snr_thr = snr_thr
        self.snr_strong_thr = snr_strong_thr
        self.prior_sigma = prior_sigma
        self.prior_floor = prior_floor

        # Hysteresis and kinematic limits
        self.lock_frames_req = lock_frames_req
        self.max_missed_frames = max_missed_frames
        self.fast_relock_frames = fast_relock_frames
        self.max_v = max_v
        self.acq_radius_sq = acq_radius_sq

        # Output declarations
        self.outputs['out_subapdata'] = self.subapdata
        self.slopes.single_mask = self.subapdata.single_mask()
        self.slopes.display_map = self.subapdata.display_map

        # --- Memory Allocation for FSM and Kinematics ---
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

        # --- Grids Generation ---
        self.x_grid = self.xp.arange(np_sub, dtype=self.dtype)
        self.y_grid = self.xp.arange(np_sub, dtype=self.dtype)
        self.xx, self.yy = self.xp.meshgrid(self.x_grid, self.y_grid)

        # --- Static Analytical Template (FFT Centered & Shifted) ---
        half_np = np_sub // 2
        dx_wrap = self.xp.where(self.x_grid > half_np - 1, self.x_grid - np_sub, self.x_grid)
        dy_wrap = self.xp.where(self.y_grid > half_np - 1, self.y_grid - np_sub, self.y_grid)
        xx_wrap, yy_wrap = self.xp.meshgrid(dx_wrap, dy_wrap)

        # Shift the template by half-pixel for even grids.
        # This forces the correlation of a perfectly centered spot to peak EXACTLY
        # on an integer index, eliminating numerical noise ambiguity in argmax.
        self.offset = 0.5 if np_sub % 2 == 0 else 0.0
        sigma = self.fwhm_pix / (2.0 * self.xp.sqrt(2.0 * self.xp.log(2.0)))
        template_centered = self.xp.exp(-((xx_wrap - self.offset)**2 + (yy_wrap - self.offset)**2) / (2 * sigma**2))
        template_centered /= self.xp.sum(template_centered)

        # Precompute the FFT of the template for extreme speed
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
        """
        Generates 2D Gaussian masks on the physical sensor grid.
        Natively handles both scalar and vector centers.
        """
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
        """
        Core vectorized GPU algorithm executing the Matched Filter and the FSM.
        """
        if self.subapdata is None:
            self.logger.warning('subapdata is not valid.')
            return

        in_pixels = self.local_inputs['in_pixels'].pixels
        n_subaps = self.nsubaps()
        np_sub = self.subapdata.np_sub

        # 0. Extract, reshape and cast pixels to ensure type consistency
        idx2d = unravel_index_2d(self.subap_idx, in_pixels.shape, self.xp)
        pixels = in_pixels[idx2d].reshape(n_subaps, np_sub, np_sub).astype(self.dtype)

        # 1. Base Spatial Correlation (The Likelihood) via FFT
        fft_pixels = self.xp.fft.fft2(pixels, axes=(1, 2))
        corr_map = self.xp.fft.ifft2(fft_pixels * self.xp.conj(self.fft_template), axes=(1, 2)).real

        # Maintain global statistics of the correlation map (the background noise floor)
        c_mean = self.xp.mean(corr_map, axis=(1, 2))
        c_std = self.xp.std(corr_map, axis=(1, 2))
        c_std_safe = self.xp.maximum(c_std, 1e-6)

        # GLOBAL SNR (The Radar): Is there a bright spot ANYWHERE in the sub-aperture?
        c_max_global = self.xp.max(corr_map, axis=(1, 2))
        snr_global = (c_max_global - c_mean) / c_std_safe
        valid_global_snr = snr_global >= self.snr_thr

        # 2. Kinematic Prediction (Swarm Tracking via Global Tip-Tilt)
        if self.xp.any(self.is_locked):
            vx_valid = self.state_x1[self.is_locked] - self.state_x2[self.is_locked]
            vy_valid = self.state_y1[self.is_locked] - self.state_y2[self.is_locked]

            # Median gracefully rejects single-subaperture outliers.
            # Clip limits the physical slew rate to avoid unrealistic DM jumps.
            v_global_x = self.xp.clip(self.xp.median(vx_valid), -self.max_v, self.max_v)
            v_global_y = self.xp.clip(self.xp.median(vy_valid), -self.max_v, self.max_v)
        else:
            v_global_x = 0.0
            v_global_y = 0.0

        x_pred = self.state_x1 + 0.5 * v_global_x
        y_pred = self.state_y1 + 0.5 * v_global_y

        # 3. Bayesian Spatial Prior (Kinematic Masking)
        # Flat prior for subapertures in Acquisition State OR when SNR is extremely high
        override_mask = (snr_global > self.snr_strong_thr) | ~self.is_locked

        prior_gauss = self._generate_gaussian(x_pred - self.offset, y_pred - self.offset, self.prior_sigma)
        spatial_prior = self.xp.where(
            override_mask[:, None, None],
            self.xp.ones_like(prior_gauss),
            (1.0 - self.prior_floor) * prior_gauss + self.prior_floor
        )

        weighted_corr = corr_map * spatial_prior

        # 4. Coarse Peak Extraction
        corr_map_flat = corr_map.reshape(n_subaps, -1)
        weighted_corr_flat = weighted_corr.reshape(n_subaps, -1)

        flat_idx = self.xp.argmax(weighted_corr_flat, axis=1)
        y_idx = flat_idx // np_sub
        x_idx = flat_idx % np_sub

        # LOCAL SNR (The Tracker Confidence): What is the SNR of the SPECIFIC peak we chose?
        # This prevents the FSM from hallucinating a lock on noise when the real star jumps out of the prior.
        c_max_local = corr_map_flat[self.xp.arange(n_subaps), flat_idx]
        snr_local = (c_max_local - c_mean) / c_std_safe
        valid_local_snr = snr_local >= self.snr_thr

        # Transform correlation indices back to true physical coordinates
        x_coarse = x_idx + self.offset
        y_coarse = y_idx + self.offset

        # 5. Fine Tracking (Dynamic WCoG)
        dynamic_weight = self._generate_gaussian(x_coarse, y_coarse, self.fwhm_pix)
        weighted_img = self.xp.maximum(pixels, 0.0) * dynamic_weight
        flux_sum = self.xp.sum(weighted_img, axis=(1, 2))

        x_est = self.xp.sum(self.xx[None, :, :] * weighted_img, axis=(1, 2)) / self.xp.maximum(flux_sum, 1e-6)
        y_est = self.xp.sum(self.yy[None, :, :] * weighted_img, axis=(1, 2)) / self.xp.maximum(flux_sum, 1e-6)

        # Fallback to coarse coordinates if flux is exactly zero
        zero_flux_mask = flux_sum < 1e-6
        x_est = self.xp.where(zero_flux_mask, x_coarse.astype(self.dtype), x_est)
        y_est = self.xp.where(zero_flux_mask, y_coarse.astype(self.dtype), y_est)

        # ========================================================
        # 6. FINITE STATE MACHINE (Confidence Matrix Logic)
        # ========================================================
        new_is_locked = self.is_locked.copy()
        new_lock_counter = self.lock_counter.copy()
        new_miss_counter = self.miss_counter.copy()
        new_state_x1 = self.state_x1.copy()
        new_state_y1 = self.state_y1.copy()

        # --- A. ACQUISITION MODE ---
        acq_mask = ~self.is_locked
        valid_acq = acq_mask & valid_local_snr
        invalid_acq = acq_mask & ~valid_local_snr

        # Calculate spatial consistency: new peak must be close to the previous one
        dist_sq = (x_est - self.state_x1)**2 + (y_est - self.state_y1)**2
        is_consistent = dist_sq <= self.acq_radius_sq
        is_first_hit = self.lock_counter == 0

        consistent_acq = valid_acq & (is_consistent | is_first_hit)
        inconsistent_acq = valid_acq & ~is_consistent

        # Logic for the lock counter
        new_lock_counter[consistent_acq] += 1
        new_lock_counter[inconsistent_acq] = 1  # Found a valid but distant peak, restart counter
        new_lock_counter[invalid_acq] = 0

        # Always update the position if a valid signal is found (consistent or not)
        new_state_x1[valid_acq] = x_est[valid_acq]
        new_state_y1[valid_acq] = y_est[valid_acq]

        lock_achieved = valid_acq & (new_lock_counter >= self.lock_frames_req)
        new_is_locked[lock_achieved] = True
        new_miss_counter[lock_achieved] = 0

        # --- B. TRACKING MODE ---
        trk_mask = self.is_locked
        valid_trk = trk_mask & valid_local_snr
        invalid_trk = trk_mask & ~valid_local_snr

        new_miss_counter[valid_trk] = 0
        new_state_x1[valid_trk] = x_est[valid_trk]
        new_state_y1[valid_trk] = y_est[valid_trk]

        # Kinematic HOLD: advance blindly via global tip-tilt prediction
        new_miss_counter[invalid_trk] += 1
        new_state_x1[invalid_trk] = x_pred[invalid_trk]
        new_state_y1[invalid_trk] = y_pred[invalid_trk]

        # --- C. DROP LOGIC (The Confidence Matrix) ---
        # Condition 1: Pure fading. Global SNR is low, Local SNR is low.
        # Action: Wait patiently up to max_missed_frames.
        slow_drop = invalid_trk & ~valid_global_snr & (new_miss_counter >= self.max_missed_frames)

        # Condition 2: Persistent Anomaly. Global SNR is high (star is there!), but Local SNR is low (outside mask).
        # Action: It's likely a DM step. Drop lock quickly to re-acquire the global peak.
        fast_drop = invalid_trk & valid_global_snr & (new_miss_counter >= self.fast_relock_frames)

        # Absolute failsafe
        hard_drop = invalid_trk & (new_miss_counter >= self.max_missed_frames)

        lock_lost = slow_drop | fast_drop | hard_drop
        new_is_locked[lock_lost] = False
        new_lock_counter[lock_lost] = 0

        # --- Update Global Memory ---
        self.state_x2 = self.state_x1.copy()
        self.state_y2 = self.state_y1.copy()
        self.state_x1 = new_state_x1
        self.state_y1 = new_state_y1
        self.is_locked = new_is_locked
        self.lock_counter = new_lock_counter
        self.miss_counter = new_miss_counter

        # --- Output Slopes to the Reconstructor ---
        # Slopec expects values relative to the nominal center AND normalized
        # to match the [-1, 1] range standard used by ShSlopec.
        cntrd = (np_sub - 1) / 2.0
        norm_factor = np_sub / 2.0

        raw_slope_x = (self.state_x1 - cntrd) / norm_factor
        raw_slope_y = (self.state_y1 - cntrd) / norm_factor

        # CRITICAL FIX: Only output slopes if the sub-aperture is firmly locked.
        # During acquisition (Locked=False), we output exactly 0.0 to prevent 
        # the DM from moving the spot and breaking the spatial consistency check.
        self.slopes.xslopes = self.xp.where(self.is_locked, raw_slope_x, 0.0)
        self.slopes.yslopes = self.xp.where(self.is_locked, raw_slope_y, 0.0)
        self.slopes.generation_time = self.current_time

        # Update telemetry
        self.flux_per_subaperture_vector.value[:] = flux_sum
        self.total_counts.value[0] = self.xp.sum(flux_sum)
        self.subap_counts.value[0] = self.xp.mean(flux_sum)

        # plot for debugging
        #debug_plot = False
        if self.t_to_seconds(self.current_time) < 1.75:
            debug_plot = False
        else:
            debug_plot = True
        if debug_plot:
            import matplotlib.pyplot as plt
            from specula import cpuArray
            subap_to_plot = 0

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Extract current state for conditional plotting
            is_locked_curr = bool(self.is_locked[subap_to_plot])

            # --- 1. Raw Pixel Image (Spot) ---
            ax = axes[0, 0]
            im1 = ax.imshow(cpuArray(pixels[subap_to_plot]), cmap='hot')
            ax.axhline(cntrd, color='w', linestyle=':', alpha=0.3)
            ax.axvline(cntrd, color='w', linestyle=':', alpha=0.3)
            ax.set_title(f'Subap {subap_to_plot}: Raw Spot (Pixels)')
            plt.colorbar(im1, ax=ax)
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            # --- 2. Correlation Map ---
            ax = axes[0, 1]
            im2 = ax.imshow(cpuArray(corr_map[subap_to_plot]), cmap='viridis')
            ax.axhline(cntrd, color='w', linestyle=':', alpha=0.3)
            ax.axvline(cntrd, color='w', linestyle=':', alpha=0.3)

            # Mark coarse peak: Empty red circle, thin edge
            ax.plot(cpuArray(x_idx[subap_to_plot]), cpuArray(y_idx[subap_to_plot]),
                    marker='o', color='red', markerfacecolor='none',
                    markersize=16, markeredgewidth=1.0, linestyle='', label='Coarse Peak')

            # Show prediction only if in Tracking: Empty cyan square, thin edge
            if is_locked_curr:
                ax.plot(cpuArray(x_pred[subap_to_plot]) - self.offset, cpuArray(y_pred[subap_to_plot]) - self.offset,
                        marker='s', color='cyan', markerfacecolor='none', 
                        markersize=16, markeredgewidth=1.0, linestyle='', label='Prediction')

            ax.set_title(f'Subap {subap_to_plot}: Correlation Map\nLocal SNR={snr_local[subap_to_plot]:.2f} | Global SNR={snr_global[subap_to_plot]:.2f}')
            plt.colorbar(im2, ax=ax)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            # --- 3. Spatial Prior (Weight Mask) ---
            ax = axes[1, 0]
            im3 = ax.imshow(cpuArray(spatial_prior[subap_to_plot]), cmap='bone', vmin=0.0, vmax=1.0)

            title_prior = "(GAUSSIAN - Tracking)" if is_locked_curr else "(FLAT - Acquisition)"
            ax.set_title(f'Subap {subap_to_plot}: Spatial Prior Mask\n{title_prior}')
            plt.colorbar(im3, ax=ax)

            if is_locked_curr:
                # Empty blue diamond for the prior
                ax.plot(cpuArray(x_pred[subap_to_plot]) - self.offset, cpuArray(y_pred[subap_to_plot]) - self.offset,
                        marker='D', color='dodgerblue', markerfacecolor='none', 
                        markersize=14, markeredgewidth=1.0, linestyle='', label='Prior Center')
                ax.legend(loc='upper right', fontsize=9)

            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            # --- 4. Weighted Image (WCoG) ---
            ax = axes[1, 1]
            im4 = ax.imshow(cpuArray(weighted_img[subap_to_plot]), cmap='hot')
            ax.axhline(cntrd, color='w', linestyle='--', alpha=0.4, linewidth=1)
            ax.axvline(cntrd, color='w', linestyle='--', alpha=0.4, linewidth=1, label='Nominal Center')

            # Mark final WCoG position: Green star
            ax.plot(cpuArray(x_est[subap_to_plot]), cpuArray(y_est[subap_to_plot]), 'g*', markersize=15, label=f'WCoG Est: ({x_est[subap_to_plot]:.2f}, {y_est[subap_to_plot]:.2f})')
            ax.set_title(f'Subap {subap_to_plot}: Weighted Image (WCoG)\nFlux={flux_sum[subap_to_plot]:.1f}')
            plt.colorbar(im4, ax=ax)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            # --- Global FSM state info ---
            fig.suptitle(f'FSM State: Locked={is_locked_curr} | Lock Counter={self.lock_counter[subap_to_plot]} | Miss Counter={self.miss_counter[subap_to_plot]}',
                         fontsize=14, fontweight='bold', color='green' if is_locked_curr else 'red')

            plt.tight_layout()
            plt.show()

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_subapdata'].generation_time = self.current_time
