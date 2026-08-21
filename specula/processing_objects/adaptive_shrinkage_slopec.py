from specula.base_processing_obj import OutputDesc
from specula.data_objects.subap_data import SubapData
from specula.lib.utils import unravel_index_2d
from specula.processing_objects.slopec import Slopec


class AdaptiveShrinkageSlopec(Slopec):
    """
    Memoryless matched-filter centroider with variance-weighted (Wiener/MMSE)
    output shrinkage, for closed-loop low-order sensing at extreme low flux.

    Design principles
    -----------------
    1. The measurement path is strictly memoryless. No temporal filtering is
       applied to pixels or to correlation maps before the position estimate.
       The only temporal memory in the measurement path is a slow EMA on the
       *scalar gain* w_t, which adds no phase to the measured signal.

    2. All three shrinkages are the same Wiener estimator applied at three
       stages, each contracting toward a different, physically meaningful prior:
         - B_reg      : contracts the WCoG toward the window centre (x_c)
         - gamma      : contracts the sub-pixel correction toward x_c
         - w_t        : contracts the emitted slope toward the loop reference (0)
       Because they multiply, the END-TO-END gain must be calibrated as a whole
       (inject a known shift, read the emitted slope, sweep magnitude) and handed
       to the control designer. Do not calibrate the stages independently.

    3. Emitting w_t -> 0 during a drop-out feeds zero error to the downstream
       integrator, which holds the DM command. This is a pure loop-gain
       reduction: it can only contract the Nyquist locus, never erode phase
       margin. Nominal IIR gain should be designed on E[w] at the reference
       magnitude, not on w = 1.

    Notes on the noise model
    ------------------------
    rho^2 (correlation SNR squared) is derived from the *detector model* and the
    measured flux, NOT from the statistics of the correlation map itself. At
    N = 1 sub-aperture, estimating the noise from the same frame that produces
    the estimate correlates w_t with the noise realisation and biases the
    effective gain (selection bias). The slow EMA on w_t is the second line of
    defence against the same effect.

    Both "flux" and its associated read-noise variance are evaluated on the SAME
    localised footprint that already forms the WCoG estimate (the Gaussian
    window centred on the coarse peak), not on the raw sub-aperture array. This
    matters whenever the sub-aperture is a large acquisition/search field, as in
    MORFEO's LO sensor (a single 240x240 px sub-aperture): summing read-noise
    over the full array would add ~n_px independent noise samples to a signal
    that only occupies a handful of pixels around the spot, driving rho^2 -> 0
    (hence w_t -> 0) unconditionally, regardless of guide star magnitude. The
    window is the same one already used for the position estimate, so this adds
    no extra branching and no extra allocation.

    Parameters
    ----------
    subapdata : SubapData
        Sub-aperture geometry and pixel indexing.
    fwhm_pix : float [pixels]
        FWHM of the nominal spot. Sets the matched-filter template.
    wcog_fwhm_pix : float [pixels] or None
        FWHM of the WCoG weighting window. Defaults to fwhm_pix. Widening it
        raises g (less contraction) at the cost of admitting more noise, and
        also widens the effective footprint used for the SNR estimate below.
    k_wiener : float [1]
        sigma_PSF^2 / sigma_s^2, where sigma_s is the closed-loop residual jitter
        RMS in pixels. w = rho^2 / (rho^2 + k_wiener). Calibrate from the error
        budget; w = 0.5 occurs at rho^2 = k_wiener.
    b_reg : float [detector units]
        Regularising pseudo-count added to the WCoG denominator. Dark-limit value
        n_sigma * sigma_e * sqrt(sum(w_p^2)), with n_sigma ~ 2-3.
    sigma_d_sq : float [pixels^2]
        Prior variance of the true offset from the integer correlation peak.
        1/12 for a uniformly distributed sub-pixel offset.
    g_wcog : float [1] or None
        Analytic WCoG gain sigma_w^2 / (sigma_s^2 + sigma_w^2). If None it is
        computed from fwhm_pix and wcog_fwhm_pix.
    excess_sq : float [1]
        Squared excess-noise factor of the detector (1.0 for a noiseless-gain
        detector, ~2.0 for EMCCD in the high-gain limit). For SPECULA's CCD
        object with excess_noise=True this is ENF^2 = 2 - 1/excess_delta.
    ron_e : float [e-]
        Read-noise standard deviation per pixel of the detector. The class
        derives the read-noise variance entering rho^2 automatically from the
        WCoG window shape (ron_e^2 * sum(window^2)): the effective number of
        noisy pixels is set by the window that already localises the position
        estimate, never by the raw sub-aperture size. Calibration scripts do
        not need to guess an "effective pixel count" any more.
    prior_sigma : float [pixels]
        Sigma of the static spatial prior, centred on the sub-aperture reference.
    prior_floor : float [1]
        Transmission floor of the spatial prior; a strong distant peak is
        penalised but never zeroed, so large excursions remain recoverable.
    w_ema_alpha : float [1]
        Smoothing factor of the EMA applied to the scalar gain w_t only.
    radar_alpha : float [1]
        Smoothing factor of the telemetry-only correlation EMA ("radar").
    snr_thr, lock_frames_req, max_missed_frames, acq_radius_sq
        Lock-declaration hysteresis. TELEMETRY ONLY: these never gate the output.
    bg_inner_radius : float [pixels] or None
        Radius beyond which pixels are used for the per-frame background
        estimate. Defaults to 0.35 * np_sub.
    stream_enable : bool [1]
        Capture calc_slopes_nofor() into a CUDA graph on GPU (see setup(),
        which calls BaseProcessingObj.build_stream()). All persistent state
        below is written in place into buffers allocated once here, never
        reassigned in calc_slopes_nofor(), which is what graph capture
        requires: only local temporaries (matched-filter/WCoG intermediates)
        are freely (re)allocated per frame, mirroring the same pattern used
        by sh.py and modulated_pyramid.py. Disable for debugging (e.g. to
        step through calc_slopes_nofor() eagerly every frame) or on CPU,
        where it is a no-op regardless (BaseProcessingObj.build_stream() only
        acts when target_device_idx >= 0).
    """

    def __init__(self,
                 subapdata: SubapData,
                 fwhm_pix: float = 1.5,
                 wcog_fwhm_pix: float = None,
                 k_wiener: float = 10.0,
                 b_reg: float = 0.0,
                 sigma_d_sq: float = 1.0 / 12.0,
                 g_wcog: float = None,
                 excess_sq: float = 1.0,
                 ron_e: float = 0.0,
                 prior_sigma: float = 5.0,
                 prior_floor: float = 0.10,
                 w_ema_alpha: float = 0.2,
                 radar_alpha: float = 0.3,
                 snr_thr: float = 3.5,
                 lock_frames_req: int = 3,
                 max_missed_frames: int = 10,
                 acq_radius_sq: float = 4.0,
                 bg_inner_radius: float = None,
                 stream_enable: bool = True,
                 **kwargs):

        self.subapdata = subapdata
        super().__init__(**kwargs)

        xp = self.xp
        n_subaps = self.subapdata.n_subaps
        np_sub = self.subapdata.np_sub
        cntrd = (np_sub - 1) / 2.0

        self.fwhm_pix = fwhm_pix
        self.wcog_fwhm_pix = fwhm_pix if wcog_fwhm_pix is None else wcog_fwhm_pix

        # --- Pre-calibrated estimator constants -----------------------------
        self.k_wiener = k_wiener
        self.b_reg = b_reg
        self.sigma_d_sq = sigma_d_sq
        self.excess_sq = excess_sq
        self.ron_e = ron_e

        sig_s = fwhm_pix / (2.0 * float(xp.sqrt(2.0 * xp.log(2.0))))
        sig_w = self.wcog_fwhm_pix / (2.0 * float(xp.sqrt(2.0 * xp.log(2.0))))
        self.sigma_psf_sq = sig_s ** 2
        self.g_wcog = (sig_w ** 2 / (sig_s ** 2 + sig_w ** 2)) if g_wcog is None else g_wcog
        self._inv_two_sig_w_sq = 1.0 / (2.0 * sig_w ** 2)

        self.prior_sigma = prior_sigma
        self.prior_floor = prior_floor
        self.w_ema_alpha = w_ema_alpha
        self.radar_alpha = radar_alpha
        self.snr_thr = snr_thr
        self.lock_frames_req = lock_frames_req
        self.max_missed_frames = max_missed_frames
        self.acq_radius_sq = acq_radius_sq

        self.cntrd = cntrd
        self.norm_factor = np_sub / 2.0
        self._eps = 1e-12
        self.stream_enable = stream_enable

        self.outputs['out_subapdata'] = self.subapdata
        self.slopes.single_mask = self.subapdata.single_mask()
        self.slopes.display_map = self.subapdata.display_map

        # --- Coordinate grids ------------------------------------------------
        grid = xp.arange(np_sub, dtype=self.dtype)
        self.xx, self.yy = xp.meshgrid(grid, grid)

        # Even-sized sub-apertures put the template origin half a pixel off the
        # integer grid; the correlation peak index i maps to position i + offset.
        self.offset = 0.5 if np_sub % 2 == 0 else 0.0

        # --- Matched-filter template (FFT-origin centred) --------------------
        half_np = np_sub // 2
        dx_wrap = xp.where(grid > half_np - 1, grid - np_sub, grid)
        xx_wrap, yy_wrap = xp.meshgrid(dx_wrap, dx_wrap)
        template = xp.exp(-((xx_wrap - self.offset) ** 2 +
                            (yy_wrap - self.offset) ** 2) / (2.0 * sig_s ** 2))
        template /= xp.sum(template)
        self.fft_template_conj = xp.conj(xp.fft.fft2(template[None, :, :], axes=(1, 2)))

        # --- Static spatial prior, centred on the loop reference -------------
        # Memoryless by construction: in closed loop the spot belongs at the
        # reference, so no predicted centre and no confirmation lock-in.
        rx = self.xx + self.offset - cntrd
        ry = self.yy + self.offset - cntrd
        prior = xp.exp(-(rx ** 2 + ry ** 2) / (2.0 * prior_sigma ** 2))
        self.spatial_prior = ((1.0 - prior_floor) * prior + prior_floor)[None, :, :].astype(self.dtype)

        # --- Effective read-noise variance for the WCoG-windowed flux --------
        # sum(window^2) is evaluated once on a window centred on the array
        # centre: the window support (a few sig_w) is negligible compared to
        # np_sub, so it is translation-invariant to numerical precision for any
        # in-bounds coarse peak, and does not need to be recomputed every frame
        # (important for CUDA graph capture: no per-frame reduction added).
        win_centred = xp.exp(-(rx ** 2 + ry ** 2) * self._inv_two_sig_w_sq)
        self._win_sumsq = float(xp.sum(win_centred ** 2))
        self.ron_var_eff = ron_e ** 2 * self._win_sumsq

        # --- Static background annulus mask ----------------------------------
        r_in = 0.35 * np_sub if bg_inner_radius is None else bg_inner_radius
        rr_sq = (self.xx - cntrd) ** 2 + (self.yy - cntrd) ** 2
        bg_mask = (rr_sq >= r_in ** 2).astype(self.dtype)
        n_bg = float(xp.maximum(xp.sum(bg_mask), 1.0))
        self.bg_mask = (bg_mask / n_bg)[None, :, :]

        # --- Persistent state -------------------------------------------------
        # w_smooth is the ONLY temporal memory in the measurement path, and it
        # carries gain, not signal.
        self.w_smooth = xp.zeros(n_subaps, dtype=self.dtype)
        self.last_x = xp.full(n_subaps, cntrd, dtype=self.dtype)
        self.last_y = xp.full(n_subaps, cntrd, dtype=self.dtype)

        # Telemetry-only radar
        self.ema_corr = xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)
        self.lock_counter = xp.zeros(n_subaps, dtype=xp.int32)
        self.miss_counter = xp.zeros(n_subaps, dtype=xp.int32)
        self.is_locked = xp.zeros(n_subaps, dtype=xp.bool_)
        self.snr_radar = xp.zeros(n_subaps, dtype=self.dtype)
        # w_out is a permanent alias of w_smooth (same buffer, set once here),
        # not a separate copy re-assigned every frame: with w_smooth written
        # in place (see calc_slopes_nofor()), this makes w_out track it for
        # free with no extra write and no CUDA-graph-unsafe re-aliasing.
        self.w_out = self.w_smooth

        # --- Pre-allocated working buffers (no allocation inside trigger) ----
        self._pix = xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)
        self._corr = xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)
        self._tmp = xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)
        self._win = xp.zeros((n_subaps, np_sub, np_sub), dtype=self.dtype)
        self._arange_n = xp.arange(n_subaps)

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

    def setup(self):
        super().setup()
        # Must run after super().setup() (which populates local_inputs via
        # get_all_inputs()): capture_stream() executes calc_slopes_nofor()
        # twice -- once eagerly to warm cupy's cuFFT plan cache (a plan
        # cannot itself be created during stream capture, see
        # BaseProcessingObj.capture_stream()), once captured into
        # self.cuda_graph -- and calc_slopes_nofor() reads
        # self.local_inputs['in_pixels'], which does not exist before
        # get_all_inputs() has run. All buffers are already sized from
        # subapdata in __init__, so there is nothing else to set up here.
        if self.stream_enable:
            super().build_stream()

    def trigger_code(self):
        self.calc_slopes_nofor()

    def calc_slopes_nofor(self):
        xp = self.xp
        n = self.nsubaps()
        np_sub = self.subapdata.np_sub
        cntrd = self.cntrd
        eps = self._eps

        # =================================================================
        # 0. Raw extraction and background removal
        # -----------------------------------------------------------------
        # The background MUST be removed before the WCoG: a constant pedestal
        # cancels in the numerator by window symmetry but inflates the
        # denominator, producing a background-dependent measurement gain.
        # No clipping at zero: negative noise excursions are part of a
        # zero-mean process and clipping them creates a positive noise
        # plateau, i.e. an uncontrolled flux-dependent shrinkage.
        # =================================================================
        in_pixels = self.local_inputs['in_pixels'].pixels
        idx2d = unravel_index_2d(self.subap_idx, in_pixels.shape, xp)
        raw = in_pixels[idx2d].reshape(n, np_sub, np_sub).astype(self.dtype)

        bg = xp.sum(raw * self.bg_mask, axis=(1, 2))
        xp.subtract(raw, bg[:, None, None], out=self._pix)
        pix = self._pix

        # True photometric flux over the full sub-aperture footprint.
        # TELEMETRY ONLY: this is deliberately NOT the flux used below for the
        # SNR / Wiener gain (see step 2) -- it exists to report what actually
        # landed on the detector, independent of the WCoG window shape.
        flux_raw = xp.sum(pix, axis=(1, 2))

        # =================================================================
        # 1. Matched filter and coarse peak under a STATIC spatial prior
        # =================================================================
        fft_pix = xp.fft.fft2(pix, axes=(1, 2))
        xp.copyto(self._corr,
                  xp.fft.ifft2(fft_pix * self.fft_template_conj, axes=(1, 2)).real)
        corr = self._corr

        xp.multiply(corr, self.spatial_prior, out=self._tmp)
        flat_idx = xp.argmax(self._tmp.reshape(n, -1), axis=1)

        y_idx = flat_idx // np_sub
        x_idx = flat_idx - y_idx * np_sub
        x_c = x_idx.astype(self.dtype) + self.offset
        y_c = y_idx.astype(self.dtype) + self.offset

        # =================================================================
        # 2. Single-pass WCoG in coordinates RELATIVE to the window centre,
        #    with a regularised denominator:
        #        x1 = x_c + sum((x - x_c) I w) / (sum(I w) + B_reg)
        # -----------------------------------------------------------------
        # Relative coordinates matter: B_reg must contract toward x_c, not
        # toward the array origin. As D -> 0 the estimate degrades smoothly to
        # x_c instead of developing a Cauchy tail.
        # =================================================================
        dx = self.xx[None, :, :] - x_c[:, None, None]
        dy = self.yy[None, :, :] - y_c[:, None, None]
        xp.exp(-(dx * dx + dy * dy) * self._inv_two_sig_w_sq, out=self._win)

        xp.multiply(pix, self._win, out=self._tmp)
        den = xp.sum(self._tmp, axis=(1, 2))
        num_x = xp.sum(self._tmp * dx, axis=(1, 2))
        num_y = xp.sum(self._tmp * dy, axis=(1, 2))

        den_reg = xp.maximum(den, 0.0) + self.b_reg + eps
        mx = num_x / den_reg
        my = num_y / den_reg

        # d_pos is the WCoG-windowed, background-subtracted flux: the same
        # localised footprint that produced mx, my above. It is what "flux"
        # means from here on -- NOT flux_raw.
        d_pos = xp.maximum(den, 0.0)

        # =================================================================
        # 3. Detector-model SNR and Wiener output gain w_t
        #        rho^2 = F^2 / (excess^2 * F + ron_var_eff)
        #        w     = rho^2 / (rho^2 + k_wiener)
        #    F = d_pos, the WCoG-windowed flux from step 2, and ron_var_eff
        #    was pre-computed at __init__ from the SAME window shape
        #    (ron_e^2 * sum(window^2)). Using the raw sub-aperture flux/pixel
        #    count here instead would sum read-noise over every pixel in the
        #    (possibly very large) acquisition footprint, not just the ones
        #    actually carrying signal -- see the class-level noise-model note.
        # -----------------------------------------------------------------
        # rho is NOT taken from the correlation-map statistics: at N = 1 that
        # would correlate w_t with the noise realisation of the same frame.
        # =================================================================
        rho_sq = d_pos * d_pos / (self.excess_sq * d_pos + self.ron_var_eff + eps)

        w_raw = rho_sq / (rho_sq + self.k_wiener)

        # The ONLY temporal filter in the measurement path, and it acts on the
        # gain, not the signal: it decorrelates w_t from the current-frame noise
        # without adding phase to the measured position.
        # In place (CUDA-graph safe): w_smooth is never reassigned, only ever
        # mutated -- see the class docstring's stream_enable note.
        self.w_smooth *= (1.0 - self.w_ema_alpha)
        self.w_smooth += self.w_ema_alpha * w_raw
        w = self.w_smooth

        # =================================================================
        # 4. Analytic single-pass grid-bias correction (replaces the iterative
        #    second WCoG pass, which amplified noise at low flux):
        #        g_eff = g * D / (D + B_reg)      (avoids double-counting the
        #                                          contraction already in B_reg)
        #        gamma = g_eff sigma_d^2 / (g_eff^2 sigma_d^2 + sigma_pos^2)
        #        x_est = x_c + gamma * (x1 - x_c)
        # -----------------------------------------------------------------
        # High SNR: gamma -> 1/g_eff, exact deconvolution, no grid bias.
        # Low  SNR: gamma -> 0, the estimate falls back to the integer peak,
        #           because sub-pixel information is simply not present.
        # =================================================================
        g_eff = self.g_wcog * d_pos / (d_pos + self.b_reg + eps)

        sigma_pos_sq = self.sigma_psf_sq / (rho_sq + eps)
        gamma = (g_eff * self.sigma_d_sq
                 / (g_eff * g_eff * self.sigma_d_sq + sigma_pos_sq + eps))

        x_est = x_c + gamma * mx
        y_est = y_c + gamma * my

        # =================================================================
        # 5. Output: purely w_t * slope. No lock gating, no hold, no clamp.
        # -----------------------------------------------------------------
        # w -> 0 feeds zero error to the downstream integrator, which holds the
        # DM command. Any hard gate here would reintroduce exactly the step
        # discontinuity this architecture exists to remove.
        # =================================================================
        slope_x = (x_est - cntrd) / self.norm_factor
        slope_y = (y_est - cntrd) / self.norm_factor

        self.slopes.xslopes = w * slope_x
        self.slopes.yslopes = w * slope_y
        self.slopes.generation_time = self.current_time

        # =================================================================
        # 6. TELEMETRY ONLY: radar EMA, lock FSM. Branch-free.
        #    Nothing below this line influences the emitted slopes.
        # -----------------------------------------------------------------
        # Every self.* update below writes in place (*=/+=/xp.copyto) into a
        # buffer allocated once in __init__, never reassigning the Python
        # attribute -- required for CUDA graph capture (see stream_enable in
        # the class docstring): only local temporaries (lock_new, miss_new,
        # locked_new, just_dropped, ...) are freely (re)computed per frame.
        # =================================================================
        self.ema_corr *= (1.0 - self.radar_alpha)
        self.ema_corr += self.radar_alpha * corr

        ema_flat = self.ema_corr.reshape(n, -1)
        ema_peak = xp.max(ema_flat, axis=1)
        ema_mean = xp.mean(ema_flat, axis=1)
        ema_std = xp.maximum(xp.std(ema_flat, axis=1), eps)
        xp.divide(ema_peak - ema_mean, ema_std, out=self.snr_radar)
        valid = self.snr_radar >= self.snr_thr

        dist_sq = (x_est - self.last_x) ** 2 + (y_est - self.last_y) ** 2
        consistent = dist_sq <= self.acq_radius_sq
        first_hit = self.lock_counter == 0

        inc = valid & (consistent | first_hit)
        lock_new = xp.where(inc, self.lock_counter + 1,
                            xp.where(valid, xp.int32(1), xp.int32(0)))
        lock_new = xp.minimum(lock_new, xp.int32(self.lock_frames_req))
        miss_new = xp.where(valid, xp.int32(0), self.miss_counter + 1)

        locked_new = xp.where(self.is_locked,
                              miss_new < self.max_missed_frames,
                              lock_new >= self.lock_frames_req)

        # Flush the radar on a genuine drop so re-acquisition sees no ghost.
        just_dropped = self.is_locked & ~locked_new
        xp.copyto(self.ema_corr, xp.where(just_dropped[:, None, None], corr, self.ema_corr))

        xp.copyto(self.lock_counter,
                  xp.where(locked_new, lock_new, xp.where(just_dropped, xp.int32(0), lock_new)))
        xp.copyto(self.miss_counter, xp.where(locked_new, miss_new, xp.int32(0)))
        xp.copyto(self.is_locked, locked_new)

        xp.copyto(self.last_x, xp.where(valid, x_est, self.last_x))
        xp.copyto(self.last_y, xp.where(valid, y_est, self.last_y))

        # True photometric flux, not the window-weighted denominator.
        # (w_out is a permanent alias of w_smooth, set once in __init__ --
        # nothing to write here, see the __init__ comment.)
        self.flux_per_subaperture_vector.value[:] = flux_raw
        self.total_counts.value[0] = xp.sum(flux_raw)
        self.subap_counts.value[0] = xp.mean(flux_raw)

    def post_trigger(self):
        super().post_trigger()
        self.outputs['out_subapdata'].generation_time = self.current_time
