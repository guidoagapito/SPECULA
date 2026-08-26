import numpy as np

from specula.base_processing_obj import OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.subap_data import SubapData
from specula.lib.make_xy import make_xy
from specula.lib.utils import unravel_index_2d
from specula.processing_objects.sh_slopec import ShSlopec


class AdaptiveWindowShSlopec(ShSlopec):
    """
        Two-step adaptive-window SH slopec processing object.

        The algorithm decouples window adaptation from slope measurement:

        1. Step 1 (slow path): an EMA of the sub-aperture image is correlated with
             a Gaussian template to estimate a coarse spot location and update the
             per-subaperture window radius.
        2. Step 2 (fast path): slopes are computed on the current frame with CoG,
             using the updated window map, and then compensated for the
             radius-dependent WCoG gain.

        Loss handling is explicit:

        - ``STATE_LOST_FADING``: low-flux fading, radius is held (no search
            expansion, to avoid injecting pure noise).
        - ``STATE_LOST_KINEMATIC``: non-fading loss, optional kinematic expansion
            towards ``max_pix_rad``.

        The dynamic window map is recomputed every frame in vectorized form
        (CUDA-graph friendly behavior).
    """

    STATE_GOOD = 0
    STATE_UNCERTAIN = 1
    STATE_LOST_FADING = 2
    STATE_LOST_KINEMATIC = 3

    def __init__(self,
                 subapdata: SubapData,
                 sn=None,
                 thr_value: float = -1,
                 exp_weight: float = 1.0,
                 filtmat=None,
                 weightedPixRad: float = 2.0,
                 windowing: bool = False,
                 weight_int_pixel_dt: float = 0,
                 window_int_pixel: bool = False,
                 window_int_threshold: float = 1.0,
                 interleave: bool = False,
                 target_device_idx: int = None,
                 precision: int = None,
                 adaptive_window_enable: bool = True,
                 ema_alpha: float = 0.2,
                 base_pix_rad: float = 2.0,
                 max_pix_rad: float = 8.0,
                 growth_gain: float = 0.7,
                 deadzone_pix: float = 0.5,
                 beta_up: float = 0.30,
                 beta_down: float = 0.10,
                 corr_snr_thr: float = 3.0,
                 flux_frac_thr: float = 0.15,
                 fading_flux_thr: float = 1.0,
                 lost_frames_req: int = 3,
                 lost_behavior_kinematic: str = 'expand',
                 template_fwhm_pix: float = 1.5,
                 psf_fwhm_pix: float = 1.5,
                 gain_comp_enable: bool = True,
                 gain_comp_min: float = 0.25,
                 gain_comp_max: float = 4.0,
                 ron_e: float = 0.0,
                 b_reg_nsigma: float = 2.5):
        """
        Parameters
        ----------
        subapdata : SubapData
            Sub-aperture geometry and indexing map.
        sn : Slopes or None, optional
            Slope-null to subtract in ``post_trigger`` (inherited behavior).
        thr_value : float, optional
            Pixel threshold value applied before CoG (see ``ShSlopec``).
        exp_weight : float, optional
            Exponent used for base CoG coordinate weights.
        filtmat : tuple or None, optional
            Optional slope filtering matrices, as in ``Slopec``.
        weightedPixRad : float, optional
            Initial SH window radius passed to ``ShSlopec``. In adaptive mode
            this acts only as startup value; dynamic radius is controlled by
            ``base_pix_rad``/``max_pix_rad`` and update rules below.
        windowing : bool, optional
            If ``True`` uses binary circular window; if ``False`` uses
            Gaussian WCoG-like weights.
        weight_int_pixel_dt : float, optional
            Integration time for internal pixel accumulation used by inherited
            weighting logic.
        window_int_pixel : bool, optional
            Enables inherited binary integration weighting mode.
        window_int_threshold : float, optional
            Threshold used by inherited ``window_int_pixel`` logic.
        interleave : bool, optional
            Interleaved slope storage flag (inherited).
        target_device_idx : int or None, optional
            Target device index (CPU/GPU backend selection).
        precision : int or None, optional
            Numeric precision selector used by SPECULA base classes.
        adaptive_window_enable : bool, optional
            Enables adaptive two-step behavior. If ``False``, slopes are
            computed with baseline ``ShSlopec`` path for compatibility.
        ema_alpha : float, optional
            EMA coefficient for Step-1 image memory. Higher values react faster,
            lower values are smoother.
        base_pix_rad : float, optional
            Minimum/nominal adaptive radius (pixels).
        max_pix_rad : float, optional
            Maximum adaptive radius (pixels).
        growth_gain : float, optional
            Gain converting coarse spot displacement into radius growth.
        deadzone_pix : float, optional
            Displacement dead-zone (pixels) before radius growth starts.
        beta_up : float, optional
            Leakage factor used when radius increases.
        beta_down : float, optional
            Leakage factor used when radius decreases.
        corr_snr_thr : float, optional
            Correlation-SNR threshold for a frame to be considered reliable.
        flux_frac_thr : float, optional
            Minimum fraction of flux inside the current window for reliability.
        fading_flux_thr : float, optional
            Flux threshold used to classify fading events.
        lost_frames_req : int, optional
            Number of consecutive bad frames required to declare persistent loss.
        lost_behavior_kinematic : {'expand', 'hold'}, optional
            Behavior during kinematic loss:
            ``'expand'`` pushes target radius to ``max_pix_rad``;
            ``'hold'`` keeps current radius.
        template_fwhm_pix : float, optional
            FWHM (pixels) of Gaussian template used for Step-1 correlation.
        psf_fwhm_pix : float, optional
            Nominal PSF FWHM (pixels) used by analytical gain compensation.
        gain_comp_enable : bool, optional
            Enables analytical compensation of radius-dependent WCoG gain.
        gain_comp_min : float, optional
            Lower clamp for gain compensation factor.
        gain_comp_max : float, optional
            Upper clamp for gain compensation factor.
        ron_e : float [e-], optional
            Read-noise standard deviation per pixel of the detector. Drives
            the CoG denominator regularisation below (0.0 = no read-noise
            floor, matching prior behaviour for a noiseless detector model;
            set to the real detector value for physical protection).
        b_reg_nsigma : float, optional
            Number of noise sigmas used to build the regularising pseudo-count
            added to the CoG denominator: ``b_reg = b_reg_nsigma * ron_e *
            sqrt(sum(window^2))``, recomputed every frame from the CURRENT
            dynamic window (``mask_weighted_dyn``), since the window radius
            -- and therefore the number of pixels effectively contributing
            noise -- changes frame to frame. This replaces a prior
            ``1/max(subap_tot, eps)`` floor that used machine epsilon
            (``np.finfo(float32).eps ~ 1.2e-7``) as its only protection: that
            is far too small to guard against real detector noise, and at
            N=1 sub-aperture (e.g. MORFEO LO) the accompanying
            ``subap_tot <= mean(subap_tot)*1e-3`` gate was additionally a
            no-op (the mean of one element is that element). Confirmed by
            direct test: pure read noise at zero flux produced spurious
            slopes up to ~22% of the full dynamic range under the old floor;
            see test_adaptive_window_sh_slopec.py.
        """

        super().__init__(subapdata=subapdata,
                         sn=sn,
                         thr_value=thr_value,
                         exp_weight=exp_weight,
                         filtmat=filtmat,
                         weightedPixRad=weightedPixRad,
                         windowing=windowing,
                         weight_int_pixel_dt=weight_int_pixel_dt,
                         window_int_pixel=window_int_pixel,
                         window_int_threshold=window_int_threshold,
                         vecWeiPixRadT=None,
                         interleave=interleave,
                         target_device_idx=target_device_idx,
                         precision=precision)

        xp = self.xp
        n_subaps = self.nsubaps()
        np_sub = self.subapdata.np_sub

        self.adaptive_window_enable = adaptive_window_enable
        self.ema_alpha = ema_alpha
        self.base_pix_rad = float(base_pix_rad)
        self.max_pix_rad = float(max_pix_rad)
        self.growth_gain = float(growth_gain)
        self.deadzone_pix = float(deadzone_pix)
        self.beta_up = float(beta_up)
        self.beta_down = float(beta_down)
        self.corr_snr_thr = float(corr_snr_thr)
        self.flux_frac_thr = float(flux_frac_thr)
        self.fading_flux_thr = float(fading_flux_thr)
        self.lost_frames_req = int(lost_frames_req)
        self.lost_behavior_kinematic = str(lost_behavior_kinematic)
        self.template_fwhm_pix = float(template_fwhm_pix)
        self.psf_fwhm_pix = float(psf_fwhm_pix)
        self.gain_comp_enable = bool(gain_comp_enable)
        self.gain_comp_min = float(gain_comp_min)
        self.gain_comp_max = float(gain_comp_max)
        self.ron_e = float(ron_e)
        self.b_reg_nsigma = float(b_reg_nsigma)

        if self.lost_behavior_kinematic not in ('expand', 'hold'):
            raise ValueError('lost_behavior_kinematic must be "expand" or "hold"')

        self._eps = np.finfo(np.float32).eps
        self._sqrt2ln2 = float(np.sqrt(2.0 * np.log(2.0)))
        self._cntrd = (np_sub - 1) / 2.0
        self._norm_factor = np_sub / 2.0
        self._offset = 0.5 if np_sub % 2 == 0 else 0.0

        # Static normalized coordinate maps (same convention as ShSlopec).
        x0, y0 = make_xy(np_sub, 1.0, xp=xp, dtype=self.dtype)
        if self.quadcell_mode:
            x0 = xp.where(x0 > 0, 1.0, -1.0)
            y0 = xp.where(y0 > 0, 1.0, -1.0)
        else:
            x0 = xp.where(x0 > 0, xp.power(x0, self.exp_weight), -xp.power(xp.abs(x0), self.exp_weight))
            y0 = xp.where(y0 > 0, xp.power(y0, self.exp_weight), -xp.power(xp.abs(y0), self.exp_weight))
        self._xweights_base = x0.astype(self.dtype)
        self._yweights_base = y0.astype(self.dtype)

        # Pixel-index grids for dynamic windows and coarse distance.
        grid = xp.arange(np_sub, dtype=self.dtype)
        self._xx, self._yy = xp.meshgrid(grid, grid)
        self._rr = xp.sqrt((self._xx - self._cntrd) ** 2 + (self._yy - self._cntrd) ** 2)

        # Correlation template (FFT centered).
        sigma_t = self.template_fwhm_pix / (2.0 * self._sqrt2ln2)
        half_np = np_sub // 2
        dx_wrap = xp.where(grid > half_np - 1, grid - np_sub, grid)
        xx_wrap, yy_wrap = xp.meshgrid(dx_wrap, dx_wrap)
        template = xp.exp(-((xx_wrap - self._offset) ** 2 + (yy_wrap - self._offset) ** 2) / (2.0 * sigma_t ** 2))
        template /= xp.sum(template)
        self._fft_template_conj = xp.conj(xp.fft.fft2(template[None, :, :], axes=(1, 2)))

        # Radius/gain state and EMA path.
        self.ema_pixels = xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)
        self.radius_curr = xp.full(n_subaps, self.base_pix_rad, dtype=self.dtype)
        self.radius_target = xp.full(n_subaps, self.base_pix_rad, dtype=self.dtype)
        self.gain_comp = xp.ones(n_subaps, dtype=self.dtype)

        # Loss-state trackers.
        self.bad_counter = xp.zeros(n_subaps, dtype=xp.int32)
        self.fading_counter = xp.zeros(n_subaps, dtype=xp.int32)
        self.state_code = xp.zeros(n_subaps, dtype=xp.int32)

        # Diagnostics.
        self.corr_snr = xp.zeros(n_subaps, dtype=self.dtype)
        self.flux_frac = xp.zeros(n_subaps, dtype=self.dtype)
        self.distance_coarse = xp.zeros(n_subaps, dtype=self.dtype)

        # Dynamic masks/weights (updated every frame in-place).
        self.mask_weighted_dyn = xp.ones((n_subaps, np_sub, np_sub), dtype=self.dtype)
        self.xweights_dyn = xp.tile(self._xweights_base[None, :, :], (n_subaps, 1, 1)).astype(self.dtype)
        self.yweights_dyn = xp.tile(self._yweights_base[None, :, :], (n_subaps, 1, 1)).astype(self.dtype)

        # Reused buffers to reduce allocations in trigger.
        self._pixels_cube = xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)

        # Reference gain used for analytical compensation.
        sig_psf = self.psf_fwhm_pix / (2.0 * self._sqrt2ln2)
        self._g_ref = 1.0
        self._sig_psf_sq = sig_psf ** 2

        self._update_dynamic_weights()

        # Telemetry outputs.
        self.radius_value = BaseValue(value=xp.copy(self.radius_curr),
                                      target_device_idx=self.target_device_idx,
                                      precision=precision)
        self.radius_target_value = BaseValue(value=xp.copy(self.radius_target),
                                             target_device_idx=self.target_device_idx,
                                             precision=precision)
        self.state_code_value = BaseValue(value=self.state_code.astype(self.dtype),
                                          target_device_idx=self.target_device_idx,
                                          precision=precision)
        self.corr_snr_value = BaseValue(value=xp.copy(self.corr_snr),
                                        target_device_idx=self.target_device_idx,
                                        precision=precision)
        self.flux_frac_value = BaseValue(value=xp.copy(self.flux_frac),
                                         target_device_idx=self.target_device_idx,
                                         precision=precision)

        self.outputs['out_radius'] = self.radius_value
        self.outputs['out_radius_target'] = self.radius_target_value
        self.outputs['out_state_code'] = self.state_code_value
        self.outputs['out_corr_snr'] = self.corr_snr_value
        self.outputs['out_flux_frac'] = self.flux_frac_value

    @classmethod
    def output_names(cls):
        result = super().output_names()
        result.update({
            'out_radius': OutputDesc(BaseValue, 'Current adaptive radius per subaperture'),
            'out_radius_target': OutputDesc(BaseValue, 'Target adaptive radius per subaperture'),
            'out_state_code': OutputDesc(BaseValue, 'Adaptive state code: 0=GOOD, 1=UNCERTAIN, 2=LOST_FADING, 3=LOST_KINEMATIC'),
            'out_corr_snr': OutputDesc(BaseValue, 'Correlation SNR proxy per subaperture'),
            'out_flux_frac': OutputDesc(BaseValue, 'Flux fraction inside current window per subaperture'),
        })
        return result

    def trigger_code(self):
        if self.weight_int_pixel_dt > 0:
            self.do_accumulation(self.current_time)
        self.calc_slopes_nofor()

    def _update_dynamic_weights(self):
        xp = self.xp

        rad = xp.clip(self.radius_curr, self.base_pix_rad, self.max_pix_rad)
        rad_3d = rad[:, None, None]

        if self.windowing:
            mask = xp.where(self._rr[None, :, :] <= rad_3d, 1.0, 0.0).astype(self.dtype)
        else:
            sigma = xp.maximum(rad_3d / self._sqrt2ln2, self._eps)
            rr_sq = (self._xx[None, :, :] - self._cntrd) ** 2 + (self._yy[None, :, :] - self._cntrd) ** 2
            mask = xp.exp(-0.5 * rr_sq / (sigma ** 2)).astype(self.dtype)
            max_mask = xp.maximum(xp.max(mask, axis=(1, 2), keepdims=True), self._eps)
            mask = mask / max_mask
            mask = xp.where(mask < 1e-6, 0.0, mask).astype(self.dtype)

        self.mask_weighted_dyn[:] = mask
        self.xweights_dyn[:] = self._xweights_base[None, :, :] * self.mask_weighted_dyn
        self.yweights_dyn[:] = self._yweights_base[None, :, :] * self.mask_weighted_dyn

    def _update_adaptive_radius(self, pixels_cube):
        xp = self.xp
        n_subaps = self.nsubaps()
        np_sub = self.subapdata.np_sub

        if not self.adaptive_window_enable:
            self.radius_target[:] = self.base_pix_rad
            self.radius_curr[:] = self.base_pix_rad
            self.state_code[:] = self.STATE_GOOD
            self.corr_snr[:] = 0
            self.flux_frac[:] = 1
            self.distance_coarse[:] = 0
            self.gain_comp[:] = 1
            self._update_dynamic_weights()
            return

        self.ema_pixels *= (1.0 - self.ema_alpha)
        self.ema_pixels += self.ema_alpha * pixels_cube

        fft_ema = xp.fft.fft2(self.ema_pixels, axes=(1, 2))
        corr = xp.fft.ifft2(fft_ema * self._fft_template_conj, axes=(1, 2)).real

        corr_flat = corr.reshape(n_subaps, -1)
        corr_peak = xp.max(corr_flat, axis=1)
        corr_mean = xp.mean(corr_flat, axis=1)
        corr_std = xp.maximum(xp.std(corr_flat, axis=1), self._eps)
        self.corr_snr[:] = (corr_peak - corr_mean) / corr_std

        flat_idx = xp.argmax(corr_flat, axis=1)
        y_idx = flat_idx // np_sub
        x_idx = flat_idx - y_idx * np_sub
        x_coarse = x_idx.astype(self.dtype) + self._offset
        y_coarse = y_idx.astype(self.dtype) + self._offset

        self.distance_coarse[:] = xp.sqrt((x_coarse - self._cntrd) ** 2 + (y_coarse - self._cntrd) ** 2)

        flux_total = xp.maximum(xp.sum(pixels_cube, axis=(1, 2)), 0.0)
        flux_in_window = xp.maximum(xp.sum(pixels_cube * self.mask_weighted_dyn, axis=(1, 2)), 0.0)
        self.flux_frac[:] = flux_in_window / (flux_total + self._eps)

        good = (self.corr_snr >= self.corr_snr_thr) & (self.flux_frac >= self.flux_frac_thr)
        fading_event = (flux_total <= self.fading_flux_thr)

        self.bad_counter[:] = xp.where(good, xp.int32(0), self.bad_counter + xp.int32(1))
        self.fading_counter[:] = xp.where(fading_event, self.fading_counter + xp.int32(1), xp.int32(0))

        lost_fading = self.fading_counter >= self.lost_frames_req
        lost_kinematic = (self.bad_counter >= self.lost_frames_req) & (~lost_fading)
        uncertain = (~good) & (~lost_fading) & (~lost_kinematic)

        self.state_code[:] = xp.where(lost_fading, self.STATE_LOST_FADING,
                                      xp.where(lost_kinematic, self.STATE_LOST_KINEMATIC,
                                               xp.where(uncertain, self.STATE_UNCERTAIN, self.STATE_GOOD))).astype(xp.int32)

        target = self.base_pix_rad + self.growth_gain * xp.maximum(self.distance_coarse - self.deadzone_pix, 0.0)
        target = xp.clip(target, self.base_pix_rad, self.max_pix_rad)

        # Critical policy: fading always holds current radius.
        # Fading must never trigger search expansion: hold the current radius
        # as soon as fading is detected, even before persistent-lost hysteresis.
        target = xp.where(fading_event, self.radius_curr, target)
        target = xp.where(lost_fading, self.radius_curr, target)
        if self.lost_behavior_kinematic == 'expand':
            target = xp.where(lost_kinematic, self.max_pix_rad, target)
        else:
            target = xp.where(lost_kinematic, self.radius_curr, target)

        self.radius_target[:] = target.astype(self.dtype)

        beta = xp.where(self.radius_target > self.radius_curr, self.beta_up, self.beta_down).astype(self.dtype)
        self.radius_curr[:] = (1.0 - beta) * self.radius_curr + beta * self.radius_target
        self.radius_curr[:] = xp.clip(self.radius_curr, self.base_pix_rad, self.max_pix_rad)

        # Analytical gain compensation for radius-dependent WCoG sensitivity.
        sig_w = xp.maximum(self.radius_curr / self._sqrt2ln2, self._eps)
        g_now = (sig_w ** 2) / (self._sig_psf_sq + sig_w ** 2 + self._eps)
        comp = self._g_ref / xp.maximum(g_now, self._eps)
        comp = xp.clip(comp, self.gain_comp_min, self.gain_comp_max)
        # gain_comp_enable is a fixed constructor-time flag (never changes
        # frame to frame), not a per-subaperture data-dependent condition --
        # a plain Python branch is correct here. xp.where(self.gain_comp_enable,
        # comp, 1.0) crashes on GPU: cupy's where() calls .astype() on the
        # condition, which a bare Python bool does not have (numpy silently
        # tolerates it, cupy does not).
        if self.gain_comp_enable:
            self.gain_comp[:] = comp
        else:
            self.gain_comp[:] = 1.0

        # CUDA-graph friendly: always recompute the full dynamic map.
        self._update_dynamic_weights()

    def calc_slopes_nofor(self):
        xp = self.xp
        n_subaps = self.nsubaps()
        np_sub = self.subapdata.np_sub

        if self.subapdata is None:
            self.logger.warning('subapdata is not valid.')
            return

        if not self.adaptive_window_enable:
            self.radius_curr[:] = self.base_pix_rad
            self.radius_target[:] = self.base_pix_rad
            self.state_code[:] = self.STATE_GOOD
            self.corr_snr[:] = 0
            self.flux_frac[:] = 1
            self.gain_comp[:] = 1

            super().calc_slopes_nofor()

            self.radius_value.value[:] = self.radius_curr
            self.radius_target_value.value[:] = self.radius_target
            self.state_code_value.value[:] = self.state_code.astype(self.dtype)
            self.corr_snr_value.value[:] = self.corr_snr
            self.flux_frac_value.value[:] = self.flux_frac
            return

        in_pixels = self.local_inputs['in_pixels'].pixels
        idx2d = unravel_index_2d(self.subap_idx, in_pixels.shape, xp)
        pixels = in_pixels[idx2d].reshape(n_subaps, np_sub, np_sub).astype(self.dtype)
        self._pixels_cube[:] = pixels

        self._update_adaptive_radius(self._pixels_cube)

        # Step 2 uses only current frame with dynamically-updated window.
        if self.thr_value > 0 and self.thr_ratio_value > 0:
            raise ValueError('Only one between _thr_value and _thr_ratio_value can be set.')

        flux_per_subaperture_vector = xp.sum(self._pixels_cube, axis=(1, 2))
        max_flux_per_subaperture = xp.max(flux_per_subaperture_vector)

        if self.thr_ratio_value > 0:
            thr = self.thr_ratio_value * max_flux_per_subaperture
        elif self.thr_pedestal or self.thr_value > 0:
            thr = self.thr_value
        else:
            thr = 0

        if self.thr_pedestal:
            pixels_thr = xp.where(self._pixels_cube < thr, 0, self._pixels_cube)
        else:
            pixels_thr = xp.maximum(self._pixels_cube - thr, 0)

        if self.store_thr_mask_cube:
            thr_mask_cube = xp.where(pixels_thr > 0, 1.0, 0.0)

        subap_tot = xp.sum(pixels_thr * self.mask_weighted_dyn, axis=(1, 2))

        # Regularised denominator, not a bare max(subap_tot, eps) floor: the
        # window radius (and therefore the pixel count feeding the CoG) is
        # dynamic, so the noise floor must be recomputed from the CURRENT
        # window every frame, not a static epsilon. b_reg is a physical
        # pseudo-count (n_sigma * read-noise * sqrt(sum(window^2))), same
        # role as AdaptiveShrinkageSlopec's B_reg. ron_e=0.0 (default)
        # intentionally reduces to no read-noise floor, matching a noiseless
        # detector model -- see the ron_e/b_reg_nsigma docstring above.
        b_reg = self.b_reg_nsigma * self.ron_e * xp.sqrt(
            xp.sum(self.mask_weighted_dyn ** 2, axis=(1, 2)))
        den_reg = xp.maximum(subap_tot, 0.0) + b_reg + self._eps
        factor = 1.0 / den_reg

        sx = xp.sum(pixels_thr * self.xweights_dyn, axis=(1, 2)) * factor
        sy = xp.sum(pixels_thr * self.yweights_dyn, axis=(1, 2)) * factor

        sx *= self.gain_comp
        sy *= self.gain_comp

        if self.mult_factor != 0:
            sx *= self.mult_factor
            sy *= self.mult_factor
            self.logger.warning('multiplication factor in the slope computer!')

        if self.store_thr_mask_cube:
            self.thr_mask_cube.value = xp.transpose(thr_mask_cube, (1, 2, 0))
            self.thr_mask_cube.generation_time = self.current_time

        self.slopes.xslopes = sx
        self.slopes.yslopes = sy
        self.slopes.generation_time = self.current_time

        self.flux_per_subaperture_vector.value[:] = flux_per_subaperture_vector
        self.total_counts.value[0] = xp.sum(flux_per_subaperture_vector)
        self.subap_counts.value[0] = xp.mean(flux_per_subaperture_vector)

        self.radius_value.value[:] = self.radius_curr
        self.radius_target_value.value[:] = self.radius_target
        self.state_code_value.value[:] = self.state_code.astype(self.dtype)
        self.corr_snr_value.value[:] = self.corr_snr
        self.flux_frac_value.value[:] = self.flux_frac

        self.logger.debug(
            f'AdaptiveWindowShSlopec slopes min/max rms: {xp.min(sx)}, {xp.max(sx)}, {xp.sqrt(xp.mean(sx ** 2))}'
        )

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_subapdata'].generation_time = self.current_time
        self.outputs['out_radius'].generation_time = self.current_time
        self.outputs['out_radius_target'].generation_time = self.current_time
        self.outputs['out_state_code'].generation_time = self.current_time
        self.outputs['out_corr_snr'].generation_time = self.current_time
        self.outputs['out_flux_frac'].generation_time = self.current_time
