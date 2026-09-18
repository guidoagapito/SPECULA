from specula.base_processing_obj import OutputDesc
from specula.base_value import BaseValue
from specula.data_objects.subap_data import SubapData
from specula.lib.confidence_gates import build_confidence_gate
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
    halo_fwhm_pix : float [pixels] or None
        FWHM of a second, broader Gaussian added to the Step-1 matched-filter
        template alongside the fwhm_pix "core" component (2026-09-13). None
        (default) leaves the template a single Gaussian, exactly as before --
        fully backward compatible. Added after finding that under degraded
        HO correction the true instantaneous spot is not a single Gaussian
        of any width: measured core+halo PSFs had only 1-25% of total flux
        within a few core pixels, the rest in a halo decaying far slower
        than any Gaussian fit to the core alone (see the 2026-09-13 HO-
        distance investigation). Widening fwhm_pix alone to compensate
        (the earlier, purely empirical fix) conflates the Step-1 template
        with the Step-2 WCoG window (wcog_fwhm_pix) and has no principled
        basis once the target is no longer Gaussian at any width -- this
        parameter lets Step-1 match the halo explicitly while fwhm_pix
        keeps describing the core (still used for sigma_psf_sq, g_wcog,
        etc.), instead of overloading fwhm_pix as a fitted compromise.
    halo_fraction : float [1]
        Fraction of the (unit-sum) matched-filter template's mass assigned
        to the halo component; the remaining (1 - halo_fraction) goes to
        the fwhm_pix core. Default 0.0: the halo contributes nothing even
        if halo_fwhm_pix is set, so the template stays a pure single
        Gaussian (both must be set to enable the two-component template).
        Only affects Step 1 (coarse peak-finding via FFT correlation) --
        Step 2's WCoG weighting (wcog_fwhm_pix) is unaffected, so the two
        can still be tuned independently as before.
    step1_fwhm_pix : float [pixels] or None
        Width of the Step-1 matched-filter template's core, decoupled from
        fwhm_pix -- None (default) falls back to fwhm_pix, bit-for-bit
        identical to before this parameter existed. Added (2026-09-15)
        after finding that Step 1's coarse-peak argmax is only as noise-
        robust as the *margin* between the winning grid hypothesis and its
        runner-up, and that margin shrinks fast as the template widens
        (noiseless margin at the loop reference: ~50% of peak height at
        fwhm_pix=1.0, ~29% at 2.0, ~8% at 4.0) -- a template narrower than
        the true PSF sharpens this margin and makes the discrete decision
        more robust to noise, independent of where the true spot sits.
        fwhm_pix itself is left untouched by this parameter and keeps
        driving sigma_psf_sq/g_wcog (Step 4's position-uncertainty prior
        and Step 2's WCoG gain) exactly as before: narrowing the Step-1
        template alone, without this decoupling, would silently also
        change those calibrated quantities and confound the margin effect
        with a change to the shrinkage prior -- the same class of mistake
        already made and corrected once for the fwhm_pix/wcog_fwhm_pix
        pair, and for halo_fwhm_pix above. NOT a fix for the pixel-
        quantization jitter documented under subpixel_peak_refine below --
        that jitter is the direct, expected consequence of a finite (not
        infinite) margin, and this parameter only widens the margin, it
        does not eliminate the underlying discreteness. subpixel_peak_refine
        was tried as a jitter fix and found unsafe (fixed 4 known-
        catastrophic seeds but regressed all 6 previously-fine seeds
        tested, 3 catastrophically) precisely because it estimates a
        continuous correction whose *sign* was found unreliable frame-to-
        frame; this parameter makes no such per-frame estimate, it only
        changes a static, precomputed template shape, so it carries none
        of that directional risk -- but its net effect on closed-loop
        resTT (margin gain vs. matched-filter SNR loss from using a
        template narrower than the true PSF) has not yet been measured
        and is not assumed to be positive.
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
    gain_correction_enable : bool [1]
        When True (default), Step 3's analytical grid-bias correction is
        applied (x_est = xc + gamma*mx, gamma->1/g_eff at high SNR). When
        False, gamma is forced to 1.0, i.e. the RAW, uncorrected WCoG
        estimate (x1 from Step 2) is emitted directly -- the geometric
        WCoG attenuation (g_wcog) is left uncompensated. Exists to
        quantify Step 3's own cost/benefit in isolation (2026-09-09).
        WARNING: disabling this changes the loop's effective open-loop
        gain (the average slope-per-unit-displacement drops by roughly
        g_eff, e.g. ~0.5-0.8 for the window sizes tested so far) --a
        closed-loop comparison with this False is only meaningful if the
        temporal_filter gain is adjusted to compensate, otherwise you are
        conflating "no correction" with "lower loop gain". Not something
        to sweep broadly: g_eff is known analytically for a given
        wcog_fwhm_pix/fwhm_pix pair, so only a couple of temporal_filter
        gains around the predicted 1/g_eff compensation are needed, not a
        blind search.
    stuck_rho_sq_thresh : float [1]
        Threshold below which a frame's rho_sq counts toward a "stuck"
        (sustained low-confidence) run (2026-09-10). Default 0.0 never
        triggers (rho_sq >= 0 always) -- both fixes below are opt-in and
        fully inert at this default, preserving prior behaviour exactly.
        Added after finding that ASHR's Design Principle 3 ("w_t -> 0
        holds the DM command, a pure loop-gain reduction, safe against a
        dropout") is NOT safe against a sustained, deterministic
        disturbance: holding the command while the true spot keeps
        moving under an ongoing tone is a runaway driven by *confidence
        collapse feeding a stuck integrator*, not a gain spike (verified
        directly via out_w_smooth/out_gamma/out_rho_sq telemetry, which
        showed effective gain drop, not rise, at a divergence onset).
        Set to a positive value (below the class's typical/nominal
        rho_sq at the operating point of interest, e.g. from
        out_rho_sq telemetry in normal operation) to arm both fixes.
    max_hold_frames : int [frames]
        Ramp length for Fix A (gain fallback, GRADUAL as of 2026-09-10):
        the emitted weight blends linearly from w_smooth toward
        fallback_w as stuck_counter goes from 0 to max_hold_frames, then
        holds at fallback_w for any longer stuck run -- not an
        all-or-nothing switch (an earlier step version was found to be
        actively harmful: a sudden full-trust jump onto a still-uncertain
        x_c made the divergence ~3x worse than doing nothing). Default
        effectively disables it (a very large ramp length, so the blend
        fraction stays ~0 for any realistic stuck run). Only meaningful
        when stuck_rho_sq_thresh > 0.
    fallback_w : float [1]
        Asymptotic weight Fix A blends toward as the stuck run lengthens,
        in place of w_smooth -- 1.0 (default) fully trusts the raw
        bias-corrected estimate at the end of the ramp (bypasses the
        shrinkage entirely) instead of continuing to emit a near-zero,
        frozen-command slope. w_smooth's own EMA state is unaffected
        (telemetry still reports the true, collapsed value); only what
        is emitted this frame is blended.
    prior_widen_factor : float [1]
        Multiplier applied to prior_sigma for a second, wider spatial
        prior that Fix B (GRADUAL as of 2026-09-10) blends toward as
        stuck_counter ramps up -- 1.0 (default) makes the wide prior
        identical to the normal one, so Fix B is a no-op regardless of
        stuck state. >1 gives the coarse-peak matched-filter search
        progressively more room to find a spot that has moved further
        from the loop reference than the normal prior tolerates well, at
        the cost of the same window-size trade-offs (g_wcog vs
        ron_var_eff, capture bistability) already documented for
        window-based methods elsewhere in this investigation -- exists
        to test whether that trade is worth it here, not assumed. An
        earlier all-or-nothing step version made the coarse peak more
        erratic the instant it switched (freer to jump to any bright
        correlation feature, not necessarily the true spot); the ramp is
        meant to let the search widen only as fast as warranted.
    prior_widen_after_frames : int [frames]
        Ramp length for Fix B: the prior used for the coarse-peak search
        blends linearly from spatial_prior toward spatial_prior_wide as
        the PREVIOUS frame's stuck_counter (this frame's own rho_sq isn't
        known until after the coarse-peak search that consumes the
        prior) goes from 0 to prior_widen_after_frames, then holds at
        spatial_prior_wide for any longer stuck run. Default effectively
        disables it. Only meaningful when stuck_rho_sq_thresh > 0.
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
    gate_type : str [1]
        Which Step-3 confidence-gate strategy to use (2026-09-18,
        replaces the earlier `relative_gate_enable` flag). One of:
        'wiener' (default) -- the classic fixed-threshold
        `rho_sq/(rho_sq+k_wiener)` gate, bit-for-bit identical to this
        class's original behaviour.
        'relative_ceiling' -- self-normalises to an EMA ceiling of the
        best recently-achievable `rho_sq` instead of a fixed constant.
        Motivation: a fixed `k_wiener` (or, equivalently, a fixed
        temporal-filter gain -- both were shown to move the same
        underlying quantity) can fix one condition while destabilising
        another, and was found NOT to give a clean, universal fix even
        for a single nominal condition across seeds (d30/H=19.5 with
        windshake: G=1.6 fixed 2 of 3 catastrophic seeds but made the
        third 2x worse). Validated only in a toy 1D closed-loop model
        and found to cleanly fix one hard real case (d30/H=19.5 seed=3)
        but be uniformly harmful for another (d55/H=19.0, because
        `margin` and `rho_sq` are not well correlated in a uniformly
        flux-starved regime) -- see RESULTS.md. Treat as experimental.
        'shifted_sigmoid' -- a 2D `min(S_snr(rho_sq), S_margin(margin))`
        gate (soft-minimum of two logistic sigmoids), toy-validated to
        improve tracking over 'wiener' across several stress regimes
        with much less chattering than a naive product-of-steep-sigmoids
        design -- see RESULTS.md, "Confidence-gate redesign" section.
        NOT yet validated in real closed loop.
        See `specula.lib.confidence_gates` for the full parameter set and
        design rationale of each gate class.
    gate_params : dict or None
        Extra keyword arguments forwarded to the chosen `gate_type`'s
        constructor (see `specula.lib.confidence_gates`). Ignored (and
        may be omitted) for the default 'wiener' gate, which only needs
        `k_wiener` above. Example for 'shifted_sigmoid':
        `{'boost_mult': 6.0, 'beta_snr': 0.5, 'margin_thresh': 0.25,
        'beta_margin': 2.0}`.
    margin_exclude_radius_px : float [pixels]
        Exclusion radius around Step 1's own coarse peak used when
        searching for the best competing correlation value elsewhere in
        the sub-aperture (excludes the peak's own shoulder, not a
        genuine distant competitor), for gate types whose
        `needs_margin` is True. Unused (and margin is not computed at
        all) for `gate_type='wiener'`.
    subpixel_peak_refine : bool [1]
        Refine Step 1's coarse peak (x_c/y_c) with a 3-point parabolic
        sub-pixel interpolation on the prior-weighted correlation map,
        instead of leaving it at the raw integer-pixel argmax. Default
        False (bit-for-bit identical to pre-2026-09-14 behaviour).
        Motivation: telemetry (out_x_c/out_y_c) showed the raw integer
        peak genuinely flickers between neighbouring pixels frame-to-
        frame even in a stable, well-tracked run (12.9% of frames at
        H=17, 59.4% at H=19.5; 69-86% of those flips reverse within 3
        frames -- noise-driven, not real target motion), since Step 2's
        WCoG window is re-centred on whichever integer pixel Step 1's
        un-refined argmax happens to land on. Parabolic interpolation
        (not log/Gaussian) is used for numerical robustness: the
        prior-weighted correlation can be near zero or (rarely, from
        noise) slightly negative, which a log-domain fit cannot handle.
        Neighbour lookup wraps periodically (`% np_sub`), matching the
        correlation's own FFT-circular topology -- not an approximation
        at the sub-aperture edge, but the mathematically consistent
        neighbour there. Falls back to zero correction (no offset) when
        the local curvature is too flat to trust relative to the peak
        value (see calc_slopes_nofor()), the same "insufficient information, don't
        guess" philosophy used elsewhere in this class (gamma -> 0 at
        low SNR). Additionally scaled by `self.w_smooth` as it stands at
        the START of the frame (last frame's EMA-smoothed Wiener weight):
        local curvature alone is not a reliable trust signal -- an
        atomic, per-seed check (2026-09-14, 4 known-catastrophic
        seeds across two triggers, old noise model, unrefined vs.
        always-on refinement) found the always-on version fixes half
        the cases (e.g. 51353nm -> 527nm) and makes the other half
        markedly worse (e.g. 42380nm -> 58211nm), a seed-dependent, not
        trigger-dependent, split -- consistent with a spurious/noise-
        driven local curvature being trusted exactly when overall
        tracking confidence is already low. Gating by w_smooth degrades
        gracefully to the un-refined integer peak whenever confidence is
        already low, using no new state or independent confidence
        signal. Only refines x_c/y_c itself -- everything downstream
        (Step 2's WCoG, Step 3's gamma correction) is unchanged code,
        just now operating on a continuous rather than integer centre.
    """

    def __init__(self,
                 subapdata: SubapData,
                 fwhm_pix: float = 1.5,
                 wcog_fwhm_pix: float = None,
                 halo_fwhm_pix: float = None,
                 halo_fraction: float = 0.0,
                 step1_fwhm_pix: float = None,
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
                 gain_correction_enable: bool = True,
                 stuck_rho_sq_thresh: float = 0.0,
                 max_hold_frames: int = 1_000_000,
                 fallback_w: float = 1.0,
                 prior_widen_factor: float = 1.0,
                 prior_widen_after_frames: int = 1_000_000,
                 stream_enable: bool = True,
                 subpixel_peak_refine: bool = False,
                 gate_type: str = 'wiener',
                 gate_params: dict = None,
                 margin_exclude_radius_px: float = 3.0,
                 **kwargs):

        self.subapdata = subapdata
        super().__init__(**kwargs)

        xp = self.xp
        n_subaps = self.subapdata.n_subaps
        np_sub = self.subapdata.np_sub
        cntrd = (np_sub - 1) / 2.0

        self.fwhm_pix = fwhm_pix
        self.wcog_fwhm_pix = fwhm_pix if wcog_fwhm_pix is None else wcog_fwhm_pix
        self.halo_fwhm_pix = halo_fwhm_pix
        self.halo_fraction = halo_fraction
        self.step1_fwhm_pix = fwhm_pix if step1_fwhm_pix is None else step1_fwhm_pix

        # --- Pre-calibrated estimator constants -----------------------------
        self.k_wiener = k_wiener
        self.b_reg = b_reg
        self.sigma_d_sq = sigma_d_sq
        self.excess_sq = excess_sq
        self.ron_e = ron_e

        # sig_s drives sigma_psf_sq/g_wcog below (the calibrated shrinkage
        # prior and WCoG gain) and must stay tied to fwhm_pix alone; the
        # Step-1 template's own core width uses the separate step1_sig,
        # which equals sig_s unless step1_fwhm_pix overrides it -- see its
        # docstring for why these must not be the same knob.
        sig_s = fwhm_pix / (2.0 * float(xp.sqrt(2.0 * xp.log(2.0))))
        step1_sig = self.step1_fwhm_pix / (2.0 * float(xp.sqrt(2.0 * xp.log(2.0))))
        sig_w = self.wcog_fwhm_pix / (2.0 * float(xp.sqrt(2.0 * xp.log(2.0))))
        self.sigma_psf_sq = sig_s ** 2
        self.g_wcog = (sig_w ** 2 / (sig_s ** 2 + sig_w ** 2)) if g_wcog is None else g_wcog
        self._inv_two_sig_w_sq = 1.0 / (2.0 * sig_w ** 2)

        self.prior_sigma = prior_sigma
        self.prior_floor = prior_floor
        self.gain_correction_enable = gain_correction_enable
        self.stuck_rho_sq_thresh = stuck_rho_sq_thresh
        self.max_hold_frames = max_hold_frames
        self.fallback_w = fallback_w
        self.prior_widen_after_frames = prior_widen_after_frames
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
        self.subpixel_peak_refine = subpixel_peak_refine

        # --- Pluggable Step-3 confidence gate (2026-09-18, see gate_type
        # docstring) -- built once here and never swapped, so dispatching
        # to it every frame (self._gate.compute(...)) is CUDA-graph safe,
        # same reasoning as the other constructor-time flags below.
        self.margin_exclude_radius_sq = margin_exclude_radius_px ** 2
        self._gate = build_confidence_gate(gate_type, xp=xp, n_subaps=n_subaps,
                                            dtype=self.dtype, k_wiener=k_wiener,
                                            gate_params=gate_params)

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
        # Two-component core+halo template (2026-09-13, see halo_fwhm_pix
        # docstring): a single Gaussian is the wrong matched filter once the
        # true spot has a non-Gaussian halo, at any fwhm_pix. Inert by
        # default (halo_fraction=0.0 or halo_fwhm_pix=None both collapse
        # this to exactly the old single-Gaussian template).
        # Core width is step1_sig (see step1_fwhm_pix docstring), not sig_s
        # directly -- identical to sig_s unless step1_fwhm_pix overrides it.
        half_np = np_sub // 2
        dx_wrap = xp.where(grid > half_np - 1, grid - np_sub, grid)
        xx_wrap, yy_wrap = xp.meshgrid(dx_wrap, dx_wrap)
        r_sq_wrap = (xx_wrap - self.offset) ** 2 + (yy_wrap - self.offset) ** 2
        core = xp.exp(-r_sq_wrap / (2.0 * step1_sig ** 2))
        core /= xp.sum(core)
        if halo_fwhm_pix is not None and halo_fraction > 0.0:
            sig_h = halo_fwhm_pix / (2.0 * float(xp.sqrt(2.0 * xp.log(2.0))))
            halo = xp.exp(-r_sq_wrap / (2.0 * sig_h ** 2))
            halo /= xp.sum(halo)
            template = (1.0 - halo_fraction) * core + halo_fraction * halo
        else:
            template = core
        self.fft_template_conj = xp.conj(xp.fft.fft2(template[None, :, :], axes=(1, 2)))

        # --- Static spatial prior, centred on the loop reference -------------
        # Memoryless by construction: in closed loop the spot belongs at the
        # reference, so no predicted centre and no confirmation lock-in.
        rx = self.xx + self.offset - cntrd
        ry = self.yy + self.offset - cntrd
        prior = xp.exp(-(rx ** 2 + ry ** 2) / (2.0 * prior_sigma ** 2))
        self.spatial_prior = ((1.0 - prior_floor) * prior + prior_floor)[None, :, :].astype(self.dtype)

        # Second, wider prior for Fix B (2026-09-10, see prior_widen_factor
        # docstring) -- precomputed once here, like spatial_prior itself,
        # and selected between per-frame with xp.where (branch-free, CUDA
        # graph safe) rather than recomputed on the fly. Identical to
        # spatial_prior at the default prior_widen_factor=1.0.
        wide_sigma = prior_sigma * prior_widen_factor
        prior_wide = xp.exp(-(rx ** 2 + ry ** 2) / (2.0 * wide_sigma ** 2))
        self.spatial_prior_wide = ((1.0 - prior_floor) * prior_wide + prior_floor)[None, :, :].astype(self.dtype)

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

        # Consecutive-stuck-frame counter for Fix A/B (2026-09-10, see
        # stuck_rho_sq_thresh docstring). Starts at 0, so both fixes are
        # inert at init even before the first rho_sq is available.
        self.stuck_counter = xp.zeros(n_subaps, dtype=xp.int32)

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

        # --- Effective-gain telemetry (2026-09-10) ----------------------------
        # w_smooth (the shrinkage weight) and gamma (the analytic bias
        # correction, Step 3) multiply directly into the emitted slope's
        # effective gain -- exposed here, not because the output-facing
        # estimator changes, to directly observe the "does effective gain
        # spike during a large-error transient?" question empirically
        # (added to investigate a suspected positive-feedback instability:
        # rho_sq scales with displacement^2, so a larger tracking error is
        # treated as more trustworthy, which could self-reinforce). rho_sq
        # itself is exposed too since it is rho_sq, not w_smooth or gamma
        # directly, that is quadratic in displacement. TELEMETRY ONLY: none
        # of these three feed back into the emitted slopes -- they are
        # written in place (CUDA-graph safe, same pattern as w_smooth/
        # ema_corr above) purely for diagnostic data_store capture.
        self.gamma_out = xp.zeros(n_subaps, dtype=self.dtype)
        self.rho_sq_out = xp.zeros(n_subaps, dtype=self.dtype)
        self.w_smooth_value = BaseValue(value=xp.copy(self.w_smooth),
                                         target_device_idx=self.target_device_idx)
        self.gamma_value = BaseValue(value=xp.copy(self.gamma_out),
                                      target_device_idx=self.target_device_idx)
        self.rho_sq_value = BaseValue(value=xp.copy(self.rho_sq_out),
                                       target_device_idx=self.target_device_idx)
        self.outputs['out_w_smooth'] = self.w_smooth_value
        self.outputs['out_gamma'] = self.gamma_value
        self.outputs['out_rho_sq'] = self.rho_sq_value

        # Step-1 coarse-peak telemetry (2026-09-14): x_c/y_c are the raw
        # integer-pixel argmax positions the Step-2 WCoG window gets
        # centred on (see Step 1 comment below) -- exposed to check
        # whether frame-to-frame flips between neighbouring pixels
        # (a true spot straddling a pixel boundary can flip the argmax
        # on noise alone) are a measurable contributor to the residual,
        # independent of core+halo/chromatic/window-size effects.
        # TELEMETRY ONLY: does not feed back into the emitted slopes.
        self.x_c_out = xp.zeros(n_subaps, dtype=self.dtype)
        self.y_c_out = xp.zeros(n_subaps, dtype=self.dtype)
        self.x_c_value = BaseValue(value=xp.copy(self.x_c_out),
                                    target_device_idx=self.target_device_idx)
        self.y_c_value = BaseValue(value=xp.copy(self.y_c_out),
                                    target_device_idx=self.target_device_idx)
        self.outputs['out_x_c'] = self.x_c_value
        self.outputs['out_y_c'] = self.y_c_value

        # Gate telemetry (2026-09-18): out_margin is zero/inert unless
        # self._gate.needs_margin; out_rho_sq_ceiling is zero/inert unless
        # the active gate exposes a 'rho_sq_ceiling' key from its own
        # telemetry() (currently only RelativeCeilingGate). TELEMETRY
        # ONLY: any persistent state the gate itself reads (e.g.
        # RelativeCeilingGate.rho_sq_ceiling) lives on the gate object --
        # these are copies exposed as data_store-capturable outputs, same
        # split as w_smooth/w_smooth_value above.
        self.margin_out = xp.zeros(n_subaps, dtype=self.dtype)
        self.rho_sq_ceiling_out = xp.zeros(n_subaps, dtype=self.dtype)
        self.margin_value = BaseValue(value=xp.copy(self.margin_out),
                                       target_device_idx=self.target_device_idx)
        self.rho_sq_ceiling_value = BaseValue(value=xp.copy(self.rho_sq_ceiling_out),
                                               target_device_idx=self.target_device_idx)
        self.outputs['out_margin'] = self.margin_value
        self.outputs['out_rho_sq_ceiling'] = self.rho_sq_ceiling_value

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
            'out_subapdata': OutputDesc(SubapData, 'Subaperture data with geometry information'),
            'out_w_smooth': OutputDesc(BaseValue, 'EMA-smoothed Wiener shrinkage weight w_t per subaperture (telemetry only, does not feed back into the emitted slope)'),
            'out_gamma': OutputDesc(BaseValue, 'Analytic grid-bias correction factor gamma per subaperture (telemetry only)'),
            'out_rho_sq': OutputDesc(BaseValue, 'Detector-model correlation SNR^2 (rho^2) per subaperture (telemetry only)'),
            'out_x_c': OutputDesc(BaseValue, 'Step-1 coarse-peak x position, integer-pixel-quantized (telemetry only)'),
            'out_y_c': OutputDesc(BaseValue, 'Step-1 coarse-peak y position, integer-pixel-quantized (telemetry only)'),
            'out_margin': OutputDesc(BaseValue, 'Confidence-gate margin per subaperture (telemetry only; zero unless the active gate_type has needs_margin=True)'),
            'out_rho_sq_ceiling': OutputDesc(BaseValue, "EMA ceiling of rho_sq per subaperture (telemetry only; unused unless gate_type='relative_ceiling')"),
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

        # Fix B, gradual version (2026-09-10 -- replaces the original hard
        # step after the step version was found to make x_c MORE erratic
        # the instant it switched, compounding badly when combined with
        # Fix A's own hard switch): blend continuously from spatial_prior
        # toward spatial_prior_wide as stuck_counter ramps from 0 to
        # prior_widen_after_frames, instead of an all-or-nothing swap --
        # in keeping with the rest of this class's continuous-shrinkage
        # design, not a hard gate. Uses last frame's stuck_counter (this
        # frame's own rho_sq isn't known until after this search).
        # Identical to self.spatial_prior whenever prior_widen_factor=1.0
        # (the default) or stuck_counter is 0, regardless of tuning.
        blend_b = xp.clip(self.stuck_counter.astype(self.dtype)
                           / max(float(self.prior_widen_after_frames), 1.0), 0.0, 1.0)
        prior_now = self.spatial_prior + blend_b[:, None, None] * (self.spatial_prior_wide - self.spatial_prior)
        xp.multiply(corr, prior_now, out=self._tmp)
        flat_idx = xp.argmax(self._tmp.reshape(n, -1), axis=1)

        y_idx = flat_idx // np_sub
        x_idx = flat_idx - y_idx * np_sub
        x_c = x_idx.astype(self.dtype) + self.offset
        y_c = y_idx.astype(self.dtype) + self.offset

        # Confidence-gate margin signal (2026-09-17, see gate_type
        # docstring): margin between Step 1's own winning peak and the
        # best OTHER value elsewhere in the same prior-weighted decision
        # surface (self._tmp -- still holding this frame's Step-1 map
        # here, not yet overwritten by Step 2 below), excluding a disk
        # around the peak's own shoulder (not a genuine distant
        # competitor). self._gate.needs_margin is a fixed constructor-
        # time property of whichever gate class was chosen, so branching
        # on it is safe for CUDA graph capture, same reasoning as
        # subpixel_peak_refine below; `margin` itself is a local
        # temporary, freely (re)computed per frame like the Step-2 WCoG
        # intermediates, not a persistent buffer. Left as None when the
        # active gate does not need it (WienerGate.compute ignores it).
        margin = None
        if self._gate.needs_margin:
            dx_peak = self.xx[None, :, :] - x_idx[:, None, None].astype(self.dtype)
            dy_peak = self.yy[None, :, :] - y_idx[:, None, None].astype(self.dtype)
            dx_wrap = xp.minimum(xp.abs(dx_peak), np_sub - xp.abs(dx_peak))
            dy_wrap = xp.minimum(xp.abs(dy_peak), np_sub - xp.abs(dy_peak))
            excluded = (dx_wrap * dx_wrap + dy_wrap * dy_wrap) <= self.margin_exclude_radius_sq
            best_val = self._tmp[self._arange_n, y_idx, x_idx]
            corr_masked = xp.where(excluded, -xp.inf, self._tmp)
            second_val = xp.max(corr_masked.reshape(n, -1), axis=1)
            best_is_zero = best_val == 0
            safe_best = xp.where(best_is_zero, 1.0, best_val)
            margin = xp.where(best_is_zero, 0.0, (best_val - second_val) / xp.abs(safe_best))

        # subpixel_peak_refine is a fixed constructor-time flag (never
        # mutated after __init__), so branching on it here is safe for
        # CUDA graph capture -- same reasoning as gain_correction_enable
        # above. 3-point parabolic sub-pixel correction on the same
        # prior-weighted map argmax was taken from (self._tmp), applied
        # independently in x and y. Neighbour lookup wraps periodically
        # (`% np_sub`), consistent with the correlation's own FFT-circular
        # topology. See the class docstring for why parabolic (not
        # log/Gaussian) and the motivating out_x_c/out_y_c telemetry.
        if self.subpixel_peak_refine:
            xm1 = (x_idx - 1) % np_sub
            xp1 = (x_idx + 1) % np_sub
            ym1 = (y_idx - 1) % np_sub
            yp1 = (y_idx + 1) % np_sub
            f0 = self._tmp[self._arange_n, y_idx, x_idx]
            f_xm1 = self._tmp[self._arange_n, y_idx, xm1]
            f_xp1 = self._tmp[self._arange_n, y_idx, xp1]
            f_ym1 = self._tmp[self._arange_n, ym1, x_idx]
            f_yp1 = self._tmp[self._arange_n, yp1, x_idx]

            # Curvature threshold is relative to the peak value itself
            # (not a fixed absolute number), so it stays meaningful
            # across very different flux/SNR levels rather than being
            # calibrated for one particular magnitude.
            min_curvature = 1e-6 * xp.abs(f0) + self._eps
            den_x = f_xm1 - 2.0 * f0 + f_xp1
            den_y = f_ym1 - 2.0 * f0 + f_yp1
            flat_enough_x = xp.abs(den_x) < min_curvature
            flat_enough_y = xp.abs(den_y) < min_curvature
            safe_den_x = xp.where(flat_enough_x, 1.0, den_x)
            safe_den_y = xp.where(flat_enough_y, 1.0, den_y)
            dx = xp.where(flat_enough_x, 0.0, 0.5 * (f_xm1 - f_xp1) / safe_den_x)
            dy = xp.where(flat_enough_y, 0.0, 0.5 * (f_ym1 - f_yp1) / safe_den_y)
            # a 3-point parabolic fit is only meaningful within the span of
            # its own 3 samples -- clip so a near-degenerate fit cannot
            # extrapolate the centre past the adjacent pixel it came from.
            # Confidence gate (2026-09-14): scale the correction by
            # self.w_smooth AS IT STANDS AT THE START OF THIS FRAME (last
            # frame's EMA-smoothed Wiener weight -- not yet overwritten by
            # Step 4/5 below, same "read before this frame's own update"
            # pattern already used for stuck_counter above). Local curvature
            # alone is not a reliable trust signal at low SNR: the flatness
            # guard above only catches a near-zero denominator, not a
            # noise-driven peak whose curvature merely LOOKS usable. Reusing
            # w_smooth needs no new state and degrades gracefully to the
            # un-refined integer peak exactly when confidence is already low
            # -- the same continuous-shrinkage philosophy as the rest of
            # this class, not a second, independent confidence mechanism.
            x_c = x_c + self.w_smooth * xp.clip(dx, -0.5, 0.5)
            y_c = y_c + self.w_smooth * xp.clip(dy, -0.5, 0.5)

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

        # Pluggable confidence gate (see gate_type docstring): self._gate
        # is fixed at construction time and never swapped, so this
        # dispatch is CUDA-graph safe (same reasoning as
        # gain_correction_enable/subpixel_peak_refine elsewhere in this
        # method). Any persistent state a gate needs (e.g.
        # RelativeCeilingGate.rho_sq_ceiling) is owned and mutated in
        # place by the gate object itself.
        w_raw = self._gate.compute(xp, rho_sq, margin, eps)

        # The ONLY temporal filter in the measurement path, and it acts on the
        # gain, not the signal: it decorrelates w_t from the current-frame noise
        # without adding phase to the measured position.
        # In place (CUDA-graph safe): w_smooth is never reassigned, only ever
        # mutated -- see the class docstring's stream_enable note.
        self.w_smooth *= (1.0 - self.w_ema_alpha)
        self.w_smooth += self.w_ema_alpha * w_raw
        w = self.w_smooth

        # Stuck-frame counter for Fix A/B (2026-09-10, see
        # stuck_rho_sq_thresh docstring): counts consecutive frames with
        # rho_sq below threshold, resetting on any frame above it. In
        # place (CUDA-graph safe). At the default threshold (0.0) this is
        # always False (rho_sq >= 0), so stuck_counter stays permanently
        # 0 and neither fix below can ever trigger.
        is_stuck_now = rho_sq < self.stuck_rho_sq_thresh
        xp.copyto(self.stuck_counter,
                  xp.where(is_stuck_now, self.stuck_counter + 1, xp.int32(0)))

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

        # gain_correction_enable is a fixed constructor-time flag (never
        # changes frame to frame), not a per-subaperture data-dependent
        # condition -- a plain Python branch is correct here, same
        # reasoning as AdaptiveWindowShSlopec's gain_comp_enable (see its
        # own comment: xp.where on a bare Python bool crashes on GPU).
        # gamma=1.0 emits x1/y1 (the raw, uncorrected Step-2 WCoG estimate)
        # directly -- see the gain_correction_enable docstring for why a
        # closed-loop comparison with this False needs the temporal_filter
        # gain adjusted to compensate, not a same-gain A/B.
        if not self.gain_correction_enable:
            gamma = 1.0

        x_est = x_c + gamma * mx
        y_est = y_c + gamma * my

        # =================================================================
        # 5. Output: purely w_t * slope, blended toward Fix A's fallback
        #    (2026-09-10, gradual version -- see max_hold_frames docstring)
        #    as the stuck run lengthens, instead of an all-or-nothing
        #    switch (found to be actively harmful in its step form: a
        #    sudden full-trust jump onto a still-uncertain x_c). At the
        #    default max_hold_frames (effectively infinite) or
        #    stuck_counter == 0, w_emit == w == w_smooth always, i.e.
        #    exactly the original "no lock gating, no hold, no clamp"
        #    behaviour.
        # -----------------------------------------------------------------
        # w -> 0 feeds zero error to the downstream integrator, which holds the
        # DM command -- safe against a random dropout, NOT safe against a
        # sustained deterministic disturbance that keeps moving the true
        # spot while frozen (see the class-level note this fix responds
        # to). Fix A bounds how long that freeze is tolerated before
        # blending toward fallback_w (default 1.0: trust the raw,
        # bias-corrected estimate fully) instead of continuing to hold.
        # w_smooth's own EMA state (and its telemetry) is untouched --
        # only what is emitted this frame changes.
        # =================================================================
        blend_a = xp.clip(self.stuck_counter.astype(self.dtype)
                           / max(float(self.max_hold_frames), 1.0), 0.0, 1.0)
        w_emit = w + blend_a * (self.fallback_w - w)

        slope_x = (x_est - cntrd) / self.norm_factor
        slope_y = (y_est - cntrd) / self.norm_factor

        self.slopes.xslopes = w_emit * slope_x
        self.slopes.yslopes = w_emit * slope_y
        self.slopes.generation_time = self.current_time

        # Effective-gain telemetry (see __init__ comment) -- in place, does
        # not affect the slopes emitted above. gamma broadcasts fine even
        # when gain_correction_enable=False forced it to the Python float
        # 1.0 above (same broadcast-assignment pattern as AWSH's
        # radius_value.value[:] = ...).
        self.gamma_out[:] = gamma
        self.rho_sq_out[:] = rho_sq
        self.w_smooth_value.value[:] = self.w_smooth
        self.gamma_value.value[:] = self.gamma_out
        self.rho_sq_value.value[:] = self.rho_sq_out
        self.x_c_out[:] = x_c
        self.y_c_out[:] = y_c
        self.x_c_value.value[:] = self.x_c_out
        self.y_c_value.value[:] = self.y_c_out
        if self._gate.needs_margin:
            self.margin_out[:] = margin
            self.margin_value.value[:] = self.margin_out
        gate_telemetry = self._gate.telemetry()
        if 'rho_sq_ceiling' in gate_telemetry:
            self.rho_sq_ceiling_out[:] = gate_telemetry['rho_sq_ceiling']
            self.rho_sq_ceiling_value.value[:] = self.rho_sq_ceiling_out

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
        self.outputs['out_w_smooth'].generation_time = self.current_time
        self.outputs['out_gamma'].generation_time = self.current_time
        self.outputs['out_rho_sq'].generation_time = self.current_time
        self.outputs['out_x_c'].generation_time = self.current_time
        self.outputs['out_y_c'].generation_time = self.current_time
        self.outputs['out_margin'].generation_time = self.current_time
        self.outputs['out_rho_sq_ceiling'].generation_time = self.current_time
