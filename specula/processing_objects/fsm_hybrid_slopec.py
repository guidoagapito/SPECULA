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
                 prior_sigma: float = 5.0,
                 prior_floor: float = 0.10,
                 lock_frames_req: int = 3,
                 max_missed_frames: int = 10,
                 **kwargs):

        self.subapdata = subapdata
        super().__init__(**kwargs)

        # Tracker physical parameters
        self.fwhm_pix = fwhm_pix
        self.snr_thr = snr_thr
        self.prior_sigma = prior_sigma
        self.prior_floor = prior_floor

        # Hysteresis logic parameters
        self.lock_frames_req = lock_frames_req
        self.max_missed_frames = max_missed_frames

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

        # Calculate Correlation SNR
        c_mean = self.xp.mean(corr_map, axis=(1, 2))
        c_std = self.xp.std(corr_map, axis=(1, 2))
        c_max = self.xp.max(corr_map, axis=(1, 2))

        c_std_safe = self.xp.maximum(c_std, 1e-6)
        snr_corr = (c_max - c_mean) / c_std_safe
        valid_snr = snr_corr >= self.snr_thr

        # 2. Kinematic Prediction (Swarm Tracking via Global Tip-Tilt)
        if self.xp.any(self.is_locked):
            vx_valid = self.state_x1[self.is_locked] - self.state_x2[self.is_locked]
            vy_valid = self.state_y1[self.is_locked] - self.state_y2[self.is_locked]

            # Median gracefully rejects single-subaperture outliers
            v_global_x = self.xp.median(vx_valid)
            v_global_y = self.xp.median(vy_valid)
        else:
            v_global_x = 0.0
            v_global_y = 0.0

        x_pred = self.state_x1 + 0.5 * v_global_x
        y_pred = self.state_y1 + 0.5 * v_global_y

        # 3. Bayesian Spatial Prior (Kinematic Masking)
        # Because the correlation map is shifted by self.offset, the prior must match its grid
        prior_gauss = self._generate_gaussian(x_pred - self.offset, y_pred - self.offset, self.prior_sigma)
        spatial_prior = (1.0 - self.prior_floor) * prior_gauss + self.prior_floor

        # Flat prior for subapertures in Acquisition State
        spatial_prior = self.xp.where(
            self.is_locked[:, None, None],
            spatial_prior,
            self.xp.ones_like(spatial_prior)
        )

        weighted_corr = corr_map * spatial_prior

        # 4. Coarse Peak Extraction
        flat_idx = self.xp.argmax(weighted_corr.reshape(n_subaps, -1), axis=1)
        y_idx = flat_idx // np_sub
        x_idx = flat_idx % np_sub

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
        # 6. FINITE STATE MACHINE (Vectorized Logical Updates)
        # ========================================================
        new_is_locked = self.is_locked.copy()
        new_lock_counter = self.lock_counter.copy()
        new_miss_counter = self.miss_counter.copy()
        new_state_x1 = self.state_x1.copy()
        new_state_y1 = self.state_y1.copy()

        # --- A. ACQUISITION MODE ---
        acq_mask = ~self.is_locked
        valid_acq = acq_mask & valid_snr
        invalid_acq = acq_mask & ~valid_snr

        new_lock_counter[valid_acq] += 1
        new_state_x1[valid_acq] = x_est[valid_acq]
        new_state_y1[valid_acq] = y_est[valid_acq]

        lock_achieved = valid_acq & (new_lock_counter >= self.lock_frames_req)
        new_is_locked[lock_achieved] = True
        new_miss_counter[lock_achieved] = 0

        new_lock_counter[invalid_acq] = 0

        # --- B. TRACKING MODE ---
        trk_mask = self.is_locked
        valid_trk = trk_mask & valid_snr
        invalid_trk = trk_mask & ~valid_snr

        new_miss_counter[valid_trk] = 0
        new_state_x1[valid_trk] = x_est[valid_trk]
        new_state_y1[valid_trk] = y_est[valid_trk]

        new_miss_counter[invalid_trk] += 1
        # Kinematic HOLD: advance blindly via global tip-tilt prediction
        new_state_x1[invalid_trk] = x_pred[invalid_trk]
        new_state_y1[invalid_trk] = y_pred[invalid_trk]

        lock_lost = invalid_trk & (new_miss_counter >= self.max_missed_frames)
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

        self.slopes.xslopes = (self.state_x1 - cntrd) / norm_factor
        self.slopes.yslopes = (self.state_y1 - cntrd) / norm_factor
        self.slopes.generation_time = self.current_time

        # Update telemetry
        self.flux_per_subaperture_vector.value[:] = flux_sum
        self.total_counts.value[0] = self.xp.sum(flux_sum)
        self.subap_counts.value[0] = self.xp.mean(flux_sum)

        # plot for debugging
        debug_plot = False
        if debug_plot:
            import matplotlib.pyplot as plt
            from specula import cpuArray
            subap_to_plot = 0

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # Estraiamo lo stato corrente per rendere il plot condizionale
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
            # Mark coarse peak
            ax.plot(cpuArray(x_idx[subap_to_plot]), cpuArray(y_idx[subap_to_plot]), 'r+', markersize=15, markeredgewidth=2, label='Coarse Peak')

            # Mostra la predizione solo se siamo in Tracking
            if is_locked_curr:
                ax.plot(cpuArray(x_pred[subap_to_plot]) - self.offset, cpuArray(y_pred[subap_to_plot]) - self.offset, 'c*', markersize=15, label='Prediction')

            ax.set_title(f'Subap {subap_to_plot}: Correlation Map\nSNR={snr_corr[subap_to_plot]:.2f} (thr={self.snr_thr})')
            plt.colorbar(im2, ax=ax)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            # --- 3. Spatial Prior (Weight Mask) ---
            ax = axes[1, 0]
            # Forziamo vmin=0.0 e vmax=1.0 per avere coerenza visiva
            im3 = ax.imshow(cpuArray(spatial_prior[subap_to_plot]), cmap='bone', vmin=0.0, vmax=1.0)

            title_prior = "(GAUSSIAN - Tracking)" if is_locked_curr else "(FLAT - Acquisition)"
            ax.set_title(f'Subap {subap_to_plot}: Spatial Prior Mask\n{title_prior}')
            plt.colorbar(im3, ax=ax)

            if is_locked_curr:
                ax.plot(cpuArray(x_pred[subap_to_plot]) - self.offset, cpuArray(y_pred[subap_to_plot]) - self.offset, 'r+', markersize=15, markeredgewidth=2, label='Prior Center')
                ax.legend(loc='upper right', fontsize=9)

            ax.set_xlabel('X [pix]')
            ax.set_ylabel('Y [pix]')

            # --- 4. Weighted Image (WCoG) ---
            ax = axes[1, 1]
            im4 = ax.imshow(cpuArray(weighted_img[subap_to_plot]), cmap='hot')
            ax.axhline(cntrd, color='w', linestyle='--', alpha=0.4, linewidth=1)
            ax.axvline(cntrd, color='w', linestyle='--', alpha=0.4, linewidth=1, label='Nominal Center')

            # Mark final WCoG position
            ax.plot(cpuArray(x_est[subap_to_plot]), cpuArray(y_est[subap_to_plot]), 'g*', markersize=20, label=f'WCoG Est: ({x_est[subap_to_plot]:.2f}, {y_est[subap_to_plot]:.2f})')
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
