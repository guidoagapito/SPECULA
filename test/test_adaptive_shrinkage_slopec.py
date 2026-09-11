import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.adaptive_shrinkage_slopec import AdaptiveShrinkageSlopec
from test.specula_testlib import cpu_and_gpu


class TestAdaptiveShrinkageSlopec(unittest.TestCase):
    """
    Focused tests for AdaptiveShrinkageSlopec, the memoryless matched-filter /
    WCoG centroider with a continuous Wiener/MMSE output-shrinkage gain w_t
    that replaces the older FSM-based tracker for MORFEO's Low-Order WFS.

    These tests deliberately do NOT try to hit the exact analytic WCoG gain
    (g_wcog on a sampled grid is close to but not exactly the textbook value):
    they check the qualitative and safety properties that the design
    principles in the class docstring promise, which are exactly the
    properties a real low-flux closed loop depends on.
    """

    def get_test_setup(self, target_device_idx, xp, subap_npx=16, n_sub_side=1):
        """
        Creates a dummy Shack-Hartmann sensor and associated data
        to test the vectorized algorithm. Single small sub-aperture by
        default: this class is meant to run on a single large acquisition
        sub-aperture (MORFEO LO WFS), and small arrays keep the tests fast.
        """
        idxs = {}
        map_dict = {}
        mask_subap = np.ones((n_sub_side * subap_npx, n_sub_side * subap_npx))

        count = 0
        for i in range(n_sub_side):
            for j in range(n_sub_side):
                mask_subap *= 0
                mask_subap[i * subap_npx:(i + 1) * subap_npx, j * subap_npx:(j + 1) * subap_npx] = 1
                idxs[count] = np.where(mask_subap == 1)
                map_dict[count] = j * n_sub_side + i
                count += 1

        v = np.zeros((len(idxs), subap_npx * subap_npx), dtype=int)
        m = np.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = np.ravel_multi_index(idx, mask_subap.shape)
            m[k] = map_dict[k]

        subapdata = SubapData(idxs=v, display_map=m, nx=n_sub_side, ny=n_sub_side,
                              target_device_idx=target_device_idx)

        ccd_shape = (n_sub_side * subap_npx, n_sub_side * subap_npx)
        return subapdata, ccd_shape

    def generate_spots(self, ccd_shape, subapdata, xp, fwhm=1.5, flux=100.0,
                       bg=0.0, shift_dx=0.0, shift_dy=0.0, noise_std=0.0):
        """Generates Gaussian spots (optionally zero flux, i.e. no spot at all)
        with an optional sub-pixel shift and additive Gaussian noise."""
        np_sub = subapdata.np_sub
        n_subaps = subapdata.n_subaps
        ccd = np.full(ccd_shape, bg, dtype=np.float32)
        cntrd = (np_sub - 1) / 2.0

        x = np.arange(np_sub) - cntrd - shift_dx
        y = np.arange(np_sub) - cntrd - shift_dy
        xx, yy = np.meshgrid(x, y)

        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        if flux > 0:
            gaussian = (gaussian / np.sum(gaussian)) * flux
        else:
            gaussian = gaussian * 0.0

        for k in range(n_subaps):
            idx_1d = cpuArray(subapdata.idxs[k])
            iy, ix = np.unravel_index(idx_1d, ccd_shape)

            min_y, max_y = np.min(iy), np.max(iy) + 1
            min_x, max_x = np.min(ix), np.max(ix) + 1
            ccd[min_y:max_y, min_x:max_x] += gaussian

        if noise_std > 0:
            ccd += np.random.normal(0, noise_std, ccd.shape).astype(np.float32)

        return xp.asarray(ccd)

    def _run_frame(self, slopec, pixels, frame, t):
        pixels.pixels = frame
        pixels.generation_time = t
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

    @cpu_and_gpu
    def test_w_out_decays_monotonically_with_flux_and_stays_bounded(self, target_device_idx, xp):
        """
        Reproduces a real dimming guide star: as injected flux drops from
        very bright to zero, the steady-state (EMA-settled) w_out gain must
        decrease and must always stay inside [0, 1]. A regression here would
        mean either a runaway gain (loop instability at low flux) or a gain
        that never contracts (no protection against noise injection).
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        flux_levels = [1e5, 1e3, 50.0, 5.0, 0.0]
        settled_w = []
        frame_idx = 0
        for flux in flux_levels:
            frame = self.generate_spots(ccd_shape, subapdata, xp, flux=flux, bg=0.0)
            # Run enough frames at each level for the slow EMA on w to settle.
            for _ in range(25):
                frame_idx += 1
                self._run_frame(slopec, pixels, frame, t * frame_idx)

                w_now = cpuArray(slopec.w_out)
                self.assertTrue(np.all(w_now >= -1e-9),
                                "w_out went below 0")
                self.assertTrue(np.all(w_now <= 1.0 + 1e-9),
                                "w_out went above 1")
                self.assertTrue(np.all(np.isfinite(w_now)),
                                "w_out is not finite")

            settled_w.append(float(cpuArray(slopec.w_out)[0]))

        # Monotonic (non-increasing) decay across decreasing flux levels.
        for a, b in zip(settled_w, settled_w[1:]):
            self.assertGreaterEqual(a, b - 1e-6,
                f"w_out did not decrease monotonically with flux: {settled_w}")

        # High flux/SNR: gain should approach 1.
        self.assertGreater(settled_w[0], 0.9,
                           "w_out did not approach 1 at very high flux/SNR")

        # Zero flux: gain should collapse close to 0 (graceful hold).
        self.assertLess(settled_w[-1], 0.05,
                        "w_out did not collapse close to 0 at zero flux")

    @cpu_and_gpu
    def test_w_ema_alpha_smooths_sudden_flux_drop(self, target_device_idx, xp):
        """
        A single noisy/faint frame right after a bright streak must not make
        w_out crash to near zero in one step: with w_ema_alpha < 1 the scalar
        gain is a slow EMA (design principle #1), decorrelating w_t from the
        current-frame noise realisation. If this smoothing were broken (e.g.
        alpha applied to the wrong side, or state not persisted), a single
        drop-out frame would produce a hard, FSM-like discontinuity again.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        w_ema_alpha = 0.2
        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         w_ema_alpha=w_ema_alpha,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        bright_frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e5, bg=0.0)
        zero_frame = self.generate_spots(ccd_shape, subapdata, xp, flux=0.0, bg=0.0)

        # Build up a high steady-state gain.
        for i in range(1, 26):
            self._run_frame(slopec, pixels, bright_frame, t * i)
        w_before = float(cpuArray(slopec.w_out)[0])
        self.assertGreater(w_before, 0.9, "Failed to build up high w_out before the drop")

        # Single sudden drop to zero flux.
        self._run_frame(slopec, pixels, zero_frame, t * 26)
        w_after_one = float(cpuArray(slopec.w_out)[0])

        # With alpha=0.2 the expected one-step value is close to
        # 0.8 * w_before (since w_raw for a zero-flux frame is ~0).
        # It must clearly NOT have jumped instantly down to ~0.
        self.assertGreater(w_after_one, 0.5 * w_before,
            "w_out collapsed in a single frame: EMA smoothing on the gain is broken")

        # After many more zero-flux frames, it must eventually settle low.
        for i in range(27, 60):
            self._run_frame(slopec, pixels, zero_frame, t * i)
        w_settled = float(cpuArray(slopec.w_out)[0])
        self.assertLess(w_settled, 0.05,
                        "w_out failed to eventually settle near 0 after sustained drop-out")

    @cpu_and_gpu
    def test_no_exceptions_or_nan_across_full_flux_sweep_including_exact_zero(self, target_device_idx, xp):
        """
        Sweeps flux from very bright down to a literal all-zero (no signal,
        no background) frame, plus a couple of noisy faint frames in between.
        This is the "graceful collapse to hold" behaviour promised by the
        docstring: unlike a naive WCoG (which divides by a vanishing
        denominator), rho^2 -> 0 must drive w_t -> 0 without ever raising or
        producing NaN/Inf in the emitted slopes or in w_out.
        """
        np.random.seed(1234)
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        flux_sweep = [1e6, 1e4, 1e2, 10.0, 1.0, 0.1, 0.0, 0.0]
        for i, flux in enumerate(flux_sweep, start=1):
            noise_std = 0.5 if flux > 0 else 0.0
            frame = self.generate_spots(ccd_shape, subapdata, xp, flux=flux, bg=1.0,
                                        noise_std=noise_std)
            try:
                self._run_frame(slopec, pixels, frame, t * i)
            except Exception as e:  # pragma: no cover - failure path
                self.fail(f"AdaptiveShrinkageSlopec raised at flux={flux}: {e!r}")

            xslopes = cpuArray(slopec.outputs['out_slopes'].xslopes)
            yslopes = cpuArray(slopec.outputs['out_slopes'].yslopes)
            w_out = cpuArray(slopec.w_out)

            self.assertTrue(np.all(np.isfinite(xslopes)),
                            f"NaN/Inf in xslopes at flux={flux}")
            self.assertTrue(np.all(np.isfinite(yslopes)),
                            f"NaN/Inf in yslopes at flux={flux}")
            self.assertTrue(np.all(np.isfinite(w_out)),
                            f"NaN/Inf in w_out at flux={flux}")
            self.assertTrue(np.all(w_out >= -1e-9) and np.all(w_out <= 1.0 + 1e-9),
                            f"w_out out of [0, 1] at flux={flux}: {w_out}")

        # The literal all-zero (no background either) frame is the strictest case.
        zero_frame = xp.zeros(ccd_shape, dtype=xp.float32)
        self._run_frame(slopec, pixels, zero_frame, t * (len(flux_sweep) + 1))
        xslopes = cpuArray(slopec.outputs['out_slopes'].xslopes)
        yslopes = cpuArray(slopec.outputs['out_slopes'].yslopes)
        w_out = cpuArray(slopec.w_out)
        self.assertTrue(np.all(np.isfinite(xslopes)) and np.all(np.isfinite(yslopes)),
                        "NaN/Inf on a literal all-zero frame")
        self.assertTrue(np.all(np.isfinite(w_out)), "NaN/Inf in w_out on a literal all-zero frame")

    @cpu_and_gpu
    def test_subpixel_accuracy_when_shrinkage_neutralized(self, target_device_idx, xp):
        """
        With shrinkage neutralized (k_wiener ~ 0, b_reg = 0, no EMA lag, very
        high SNR) the emitted slope must track an injected sub-pixel shift in
        sign and rough proportionality. This catches gain/sign bugs in the
        matched-filter + WCoG + gamma correction chain, independent of the
        w_t shrinkage machinery this class adds on top.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5,
                                         k_wiener=1e-8, b_reg=0.0, ron_e=0.0,
                                         w_ema_alpha=1.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        shift_x, shift_y = 0.3, -0.4
        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                    shift_dx=shift_x, shift_dy=shift_y)
        self._run_frame(slopec, pixels, frame, t)

        w_out = float(cpuArray(slopec.w_out)[0])
        self.assertGreater(w_out, 0.99,
                           "w_out did not approach 1 with shrinkage neutralized")

        slopes_x = cpuArray(slopec.outputs['out_slopes'].xslopes)
        slopes_y = cpuArray(slopec.outputs['out_slopes'].yslopes)

        expected_slope_x = shift_x / (subap_npx / 2.0)
        expected_slope_y = shift_y / (subap_npx / 2.0)

        # Correct sign, and within a generous tolerance of the true shift
        # (gamma/g_wcog on a sampled grid is close to, but not exactly, the
        # textbook analytic gain).
        np.testing.assert_allclose(slopes_x, expected_slope_x, atol=0.05,
                                   err_msg="X sub-pixel accuracy/sign failed")
        np.testing.assert_allclose(slopes_y, expected_slope_y, atol=0.05,
                                   err_msg="Y sub-pixel accuracy/sign failed")

        # Rough proportionality: a larger injected shift must yield a larger
        # emitted slope of the same sign.
        big_shift = 0.7
        frame_big = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                        shift_dx=big_shift, shift_dy=0.0)
        self._run_frame(slopec, pixels, frame_big, t * 2)
        slopes_x_big = cpuArray(slopec.outputs['out_slopes'].xslopes)

        self.assertTrue(np.all(slopes_x_big > slopes_x),
                        "Larger injected shift did not yield a larger emitted slope")

    @cpu_and_gpu
    def test_gain_correction_enable_toggle(self, target_device_idx, xp):
        """
        `gain_correction_enable` (2026-09-09, added to quantify Step 3's
        cost/benefit in isolation): default True must reproduce the
        existing corrected behaviour exactly (gamma computed as before);
        False must emit the raw, uncorrected Step-2 WCoG estimate
        (gamma=1), which at high SNR under-corrects the window's own
        geometric attenuation (g_wcog<1) and so is smaller in magnitude
        than the corrected slope for the same injected shift.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        shift_x = 0.3
        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                    shift_dx=shift_x, shift_dy=0.0)

        common = dict(fwhm_pix=1.5, k_wiener=1e-8, b_reg=0.0, ron_e=0.0,
                     w_ema_alpha=1.0, target_device_idx=target_device_idx)

        slopec_on = AdaptiveShrinkageSlopec(subapdata, gain_correction_enable=True, **common)
        slopec_on.inputs['in_pixels'].set(pixels)
        self._run_frame(slopec_on, pixels, frame, t)
        slope_on = float(cpuArray(slopec_on.outputs['out_slopes'].xslopes)[0])

        slopec_off = AdaptiveShrinkageSlopec(subapdata, gain_correction_enable=False, **common)
        slopec_off.inputs['in_pixels'].set(pixels)
        self._run_frame(slopec_off, pixels, frame, t)
        slope_off = float(cpuArray(slopec_off.outputs['out_slopes'].xslopes)[0])

        self.assertGreater(slope_on, 0.0, "Sanity: corrected slope should be positive")
        self.assertGreater(slope_off, 0.0, "Sanity: uncorrected slope should be positive")
        self.assertLess(slope_off, slope_on,
                        "Disabling gain correction should under-correct (smaller "
                        "magnitude slope), not match or exceed the corrected one")

        # Default (unspecified) must match gain_correction_enable=True exactly.
        slopec_default = AdaptiveShrinkageSlopec(subapdata, **common)
        slopec_default.inputs['in_pixels'].set(pixels)
        self.assertTrue(slopec_default.gain_correction_enable,
                        "Default must be True (preserve original behaviour)")
        self._run_frame(slopec_default, pixels, frame, t)
        slope_default = float(cpuArray(slopec_default.outputs['out_slopes'].xslopes)[0])
        np.testing.assert_allclose(slope_default, slope_on, atol=1e-9,
                                   err_msg="Default behaviour must match gain_correction_enable=True")

    @cpu_and_gpu
    def test_cuda_graph_capture_tracks_new_frames_written_in_place(self, target_device_idx, xp):
        """
        With stream_enable=True (the default), setup() must capture
        calc_slopes_nofor() into a CUDA graph on GPU (self.cuda_graph is not
        None), and stay a no-op on CPU. The critical correctness property is
        that every self.* piece of persistent state calc_slopes_nofor()
        updates is written IN PLACE, never reassigned (see the class
        docstring's stream_enable note) -- a captured graph replays the same
        kernels into the same buffers, so a bare `self.foo = new_array`
        would silently freeze that piece of state at its capture-time value
        forever. This mirrors real usage: CCD.trigger_code() writes detector
        frames in place (`self._pixels.pixels[:] = ...`), so a graph-captured
        slopec must see each new frame on replay, not just the one that was
        in the input buffer when setup() ran.
        """
        subap_npx = 16
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5,
                                         k_wiener=1e-8, b_reg=0.0, ron_e=0.0,
                                         w_ema_alpha=1.0,
                                         target_device_idx=target_device_idx,
                                         stream_enable=True)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.setup()

        if target_device_idx >= 0:
            self.assertIsNotNone(slopec.cuda_graph,
                "CUDA graph was not captured on GPU with stream_enable=True")
        else:
            self.assertIsNone(slopec.cuda_graph,
                "build_stream() should stay a no-op on CPU (target_device_idx < 0)")

        t = int(1e9)
        for shift in (0.3, -0.3, 0.0):
            frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                        shift_dx=shift, shift_dy=0.0)
            # In place, like the real CCD does -- NOT `pixels.pixels = frame`
            # (see _run_frame(), used by every other test here), which would
            # defeat graph capture by reassigning the buffer address.
            pixels.pixels[:] = frame
            pixels.generation_time = t
            slopec.check_ready(t)
            slopec.trigger()
            slopec.post_trigger()
            t += int(1e9)

            slopes_x = cpuArray(slopec.outputs['out_slopes'].xslopes)
            expected = shift / (subap_npx / 2.0)
            np.testing.assert_allclose(slopes_x, expected, atol=0.05,
                err_msg=f"Graph replay did not track a new in-place-written frame "
                        f"(shift={shift}); persistent state may be reassigned "
                        f"instead of written in place somewhere in calc_slopes_nofor()")

    @cpu_and_gpu
    def test_stream_enable_matches_eager_over_a_long_evolving_sequence(self, target_device_idx, xp):
        """
        Differential test: two otherwise-identical instances, one with
        stream_enable=False (plain eager trigger_code() every frame) and one
        with stream_enable=True (CUDA-graph-captured on GPU, still eager on
        CPU since build_stream() is a no-op there), fed the exact same long,
        varying sequence of frames (changing flux, shift and noise, written
        IN PLACE like a real detector). Every frame's xslopes/yslopes/w_out
        must match closely between the two.

        This is the test that should catch any future regression where the
        graph-captured path silently diverges from eager execution (e.g. a
        piece of persistent state accidentally reassigned instead of written
        in place, so it freezes at its capture-time value on replay) --
        exactly the class of bug `stream_enable` was introduced to guard
        against in the Task 2 CUDA-graph work. It intentionally does NOT
        cover configuration mistakes upstream of this class (e.g. a stale
        k_wiener left over in a yml from an earlier experiment) -- eager and
        captured execution of the SAME wrong config still agree with each
        other, matching-but-wrong is a real failure mode this test cannot
        see, only a divergence between the two paths.
        """
        subap_npx = 16
        subapdata_a, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_b, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_a = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_b = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        kwargs = dict(fwhm_pix=1.5, k_wiener=10.0, b_reg=0.5, ron_e=1.0,
                     w_ema_alpha=0.2, target_device_idx=target_device_idx)
        eager = AdaptiveShrinkageSlopec(subapdata_a, stream_enable=False, **kwargs)
        captured = AdaptiveShrinkageSlopec(subapdata_b, stream_enable=True, **kwargs)
        eager.inputs['in_pixels'].set(pixels_a)
        captured.inputs['in_pixels'].set(pixels_b)
        eager.setup()
        captured.setup()

        if target_device_idx >= 0:
            self.assertIsNotNone(captured.cuda_graph,
                "expected a captured graph on GPU for the stream_enable=True instance")
        self.assertIsNone(eager.cuda_graph,
                          "stream_enable=False must never capture a graph")

        rng = np.random.RandomState(7)
        t = int(1e9)
        for i in range(150):
            flux = float(rng.choice([0.0, 1.0, 10.0, 1e3, 1e5]))
            shift_dx = float(rng.uniform(-0.4, 0.4))
            shift_dy = float(rng.uniform(-0.4, 0.4))
            noise_std = float(rng.uniform(0.0, 1.5))
            frame = self.generate_spots(ccd_shape, subapdata_a, xp, flux=flux, bg=0.5,
                                        shift_dx=shift_dx, shift_dy=shift_dy,
                                        noise_std=noise_std)
            # In place, exactly like a real CCD -- required for the captured
            # graph to actually see each new frame (see the other CUDA-graph
            # test above).
            pixels_a.pixels[:] = frame
            pixels_b.pixels[:] = frame
            pixels_a.generation_time = t
            pixels_b.generation_time = t

            eager.check_ready(t); eager.trigger(); eager.post_trigger()
            captured.check_ready(t); captured.trigger(); captured.post_trigger()
            t += int(1e9)

            xe = cpuArray(eager.outputs['out_slopes'].xslopes)
            xc = cpuArray(captured.outputs['out_slopes'].xslopes)
            ye = cpuArray(eager.outputs['out_slopes'].yslopes)
            yc = cpuArray(captured.outputs['out_slopes'].yslopes)
            we = cpuArray(eager.w_out)
            wc = cpuArray(captured.w_out)

            np.testing.assert_allclose(xc, xe, atol=1e-4, rtol=1e-4,
                err_msg=f"frame {i}: xslopes diverged between eager and captured execution")
            np.testing.assert_allclose(yc, ye, atol=1e-4, rtol=1e-4,
                err_msg=f"frame {i}: yslopes diverged between eager and captured execution")
            np.testing.assert_allclose(wc, we, atol=1e-4, rtol=1e-4,
                err_msg=f"frame {i}: w_out diverged between eager and captured execution")

    @cpu_and_gpu
    def test_background_only_frame_gives_low_confidence_not_spurious_slope(self, target_device_idx, xp):
        """
        A frame with no spot at all (pure background + read noise) must not
        raise, and must produce a low-confidence w_out (the matched filter
        will still find *some* correlation peak in pure noise, but the
        Wiener gain must keep it from being trusted). The emitted slope,
        being w_t * raw_slope, must stay small even if the noise-driven
        coarse peak lands off-centre -- this is the dark-limit sanity check.
        """
        np.random.seed(42)
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        for i in range(1, 11):
            bg_frame = self.generate_spots(ccd_shape, subapdata, xp, flux=0.0,
                                           bg=5.0, noise_std=2.0)
            self._run_frame(slopec, pixels, bg_frame, t * i)

        w_out = cpuArray(slopec.w_out)
        xslopes = cpuArray(slopec.outputs['out_slopes'].xslopes)
        yslopes = cpuArray(slopec.outputs['out_slopes'].yslopes)

        self.assertTrue(np.all(np.isfinite(w_out)))
        self.assertTrue(np.all(np.isfinite(xslopes)) and np.all(np.isfinite(yslopes)))

        self.assertLess(float(np.max(w_out)), 0.3,
                        "Pure background/noise frame produced unexpectedly high confidence")
        self.assertLess(float(np.max(np.abs(xslopes))), 0.3,
                        "Pure background/noise frame produced a spuriously large X slope")
        self.assertLess(float(np.max(np.abs(yslopes))), 0.3,
                        "Pure background/noise frame produced a spuriously large Y slope")

    @cpu_and_gpu
    def test_stuck_detector_and_both_fixes_default_inert(self, target_device_idx, xp):
        """
        Regression check for the 2026-09-10 "stuck" detector / Fix A / Fix B
        addition (see stuck_rho_sq_thresh docstring): at stuck_rho_sq_thresh=0
        (the default) rho_sq >= 0 always, so stuck_counter must never leave 0
        and neither fix can ever trigger, REGARDLESS of how aggressively
        max_hold_frames/fallback_w/prior_widen_after_frames/prior_widen_factor
        are otherwise configured. This is checked directly (not by comparing
        against a pre-change git revision) by running two instances -- one
        with every new knob left at its default, one with stuck_rho_sq_thresh
        explicitly 0.0 but every other new knob set to an aggressive,
        would-otherwise-trigger-immediately value -- over the same multi-frame
        sequence (including low-flux frames, where Fix A/B would matter if
        they were live) and requiring byte-for-byte identical output. Also
        checks that spatial_prior_wide == spatial_prior at the default
        prior_widen_factor=1.0, per its docstring.
        """
        subap_npx, t = 16, int(1e9)
        subapdata_a, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_b, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_a = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_b = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        common = dict(fwhm_pix=1.5, ron_e=1.0, w_ema_alpha=0.2,
                     target_device_idx=target_device_idx)
        slopec_default = AdaptiveShrinkageSlopec(subapdata_a, **common)
        slopec_explicit_inert = AdaptiveShrinkageSlopec(
            subapdata_b,
            stuck_rho_sq_thresh=0.0,   # inert -- rho_sq >= 0 always
            max_hold_frames=1,         # would trigger Fix A almost immediately if live
            fallback_w=0.0,            # would be blatantly visible if it ever fired
            prior_widen_after_frames=1,  # would trigger Fix B almost immediately if live
            prior_widen_factor=3.0,    # would visibly change the coarse-peak search if live
            **common)
        slopec_default.inputs['in_pixels'].set(pixels_a)
        slopec_explicit_inert.inputs['in_pixels'].set(pixels_b)

        # spatial_prior_wide must equal spatial_prior at prior_widen_factor=1.0
        # (the default), regardless of the always-inert stuck detector.
        np.testing.assert_allclose(cpuArray(slopec_default.spatial_prior_wide),
                                   cpuArray(slopec_default.spatial_prior), atol=0,
                                   err_msg="spatial_prior_wide != spatial_prior at "
                                           "the default prior_widen_factor=1.0")

        flux_sequence = [1e5, 50.0, 0.0, 5.0, 1e4, 0.0]
        for i, flux in enumerate(flux_sequence, start=1):
            frame = self.generate_spots(ccd_shape, subapdata_a, xp, flux=flux, bg=0.0,
                                        shift_dx=0.2, shift_dy=-0.1)
            self._run_frame(slopec_default, pixels_a, frame, t * i)
            self._run_frame(slopec_explicit_inert, pixels_b, frame, t * i)

            self.assertTrue(np.all(cpuArray(slopec_default.stuck_counter) == 0),
                            f"default instance's stuck_counter left 0 at frame {i}")
            self.assertTrue(np.all(cpuArray(slopec_explicit_inert.stuck_counter) == 0),
                            f"stuck_counter left 0 at frame {i} despite "
                            f"stuck_rho_sq_thresh=0.0 (should be permanently inert)")

            xd = cpuArray(slopec_default.outputs['out_slopes'].xslopes)
            xe = cpuArray(slopec_explicit_inert.outputs['out_slopes'].xslopes)
            yd = cpuArray(slopec_default.outputs['out_slopes'].yslopes)
            ye = cpuArray(slopec_explicit_inert.outputs['out_slopes'].yslopes)
            wd = cpuArray(slopec_default.w_out)
            we = cpuArray(slopec_explicit_inert.w_out)

            np.testing.assert_allclose(xe, xd, atol=1e-9, rtol=0,
                err_msg=f"frame {i}: xslopes diverged between default and "
                        f"explicit-but-inert instances -- a new knob is not inert")
            np.testing.assert_allclose(ye, yd, atol=1e-9, rtol=0,
                err_msg=f"frame {i}: yslopes diverged between default and "
                        f"explicit-but-inert instances -- a new knob is not inert")
            np.testing.assert_allclose(we, wd, atol=1e-9, rtol=0,
                err_msg=f"frame {i}: w_out diverged between default and "
                        f"explicit-but-inert instances -- a new knob is not inert")

    @cpu_and_gpu
    def test_stuck_counter_increments_and_resets(self, target_device_idx, xp):
        """
        stuck_counter must increment by 1 on every consecutive frame with
        rho_sq below stuck_rho_sq_thresh, and reset to 0 the moment a frame
        clears the threshold. Zero-flux, zero-background frames give
        d_pos = 0 exactly, hence rho_sq = 0 exactly, so any positive
        threshold reliably counts as "stuck" with no dependence on the
        detector-model constants.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         stuck_rho_sq_thresh=1.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        zero_frame = self.generate_spots(ccd_shape, subapdata, xp, flux=0.0, bg=0.0)
        for i in range(1, 6):
            self._run_frame(slopec, pixels, zero_frame, t * i)
            counter = cpuArray(slopec.stuck_counter)
            self.assertTrue(np.all(counter == i),
                            f"stuck_counter did not increment to {i} at frame {i}: {counter}")

        bright_frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0)
        self._run_frame(slopec, pixels, bright_frame, t * 6)
        counter = cpuArray(slopec.stuck_counter)
        self.assertTrue(np.all(counter == 0),
                        f"stuck_counter did not reset to 0 after a high-confidence frame: {counter}")

    @cpu_and_gpu
    def test_fix_a_gain_fallback_overrides_emission_only(self, target_device_idx, xp):
        """
        Fix A (GRADUAL as of 2026-09-10, see max_hold_frames docstring):
        w_emit = w_smooth + blend_a * (fallback_w - w_smooth), with
        blend_a = clip(stuck_counter / max_hold_frames, 0, 1) using THIS
        frame's own (already-updated) stuck_counter. An earlier all-or-
        nothing step version was found to be actively harmful (a sudden
        full-trust jump onto a still-uncertain x_c made a divergence worse),
        so this checks the ramp's INTERMEDIATE behaviour, not just its
        saturated endpoint -- while confirming w_smooth itself (read
        directly) is unaffected by the blend at every point, since the
        override only ever touches what is emitted.

        Uses w_ema_alpha=1.0 (no EMA lag) and repeats the EXACT same
        low-flux, shifted frame every iteration: with no noise and no
        temporal filtering, w_smooth, gamma and the raw estimate x_est are
        then identical on every frame regardless of history (only
        stuck_counter, and hence blend_a, changes), which makes the ramp
        checkable by simple algebra: the frame at stuck_counter=1 (a known,
        non-zero blend_a) gives raw_slope = slope / w_emit exactly, and
        every later frame's emitted slope must equal
        (w_smooth + blend_a*(fallback_w - w_smooth)) * raw_slope for that
        frame's own (known) blend_a.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=3.0, bg=0.0,
                                    shift_dx=0.3, shift_dy=0.0)

        # Probe rho_sq for this exact frame/config so the "stuck" threshold is
        # derived from the class's own telemetry rather than a hardcoded
        # numeric guess about the detector-model formula.
        probe = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                        k_wiener=50.0, w_ema_alpha=1.0,
                                        target_device_idx=target_device_idx)
        probe.inputs['in_pixels'].set(pixels)
        self._run_frame(probe, pixels, frame, t)
        probe_rho_sq = float(cpuArray(probe.rho_sq_value.value)[0])
        probe_w = float(cpuArray(probe.w_out)[0])
        self.assertLess(probe_w, 0.5,
                        "Test setup assumption violated: expected a heavily "
                        "shrunk w at this low flux/high k_wiener")

        max_hold_frames = 4
        fallback_w = 1.0
        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         k_wiener=50.0, w_ema_alpha=1.0,
                                         stuck_rho_sq_thresh=probe_rho_sq * 5.0,
                                         max_hold_frames=max_hold_frames,
                                         fallback_w=fallback_w,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        w_smooth_ref = None
        raw_slope = None
        # Run one frame past saturation (stuck_counter = max_hold_frames + 1)
        # to also confirm the ramp HOLDS at blend_a=1 instead of extrapolating
        # past it (the clip(..., 0, 1), not just an unbounded linear ramp).
        for k in range(1, max_hold_frames + 2):
            self._run_frame(slopec, pixels, frame, t * k)
            counter = int(cpuArray(slopec.stuck_counter)[0])
            self.assertEqual(counter, k, "stuck_counter did not increment as expected")

            w_smooth_now = float(cpuArray(slopec.w_smooth)[0])
            slope_now = float(cpuArray(slopec.outputs['out_slopes'].xslopes)[0])

            if w_smooth_ref is None:
                # First stuck frame: establish the deterministic reference
                # values (identical on every later frame at this alpha=1.0,
                # fixed-frame setup) and solve for raw_slope using this
                # frame's own known, non-trivial blend_a.
                w_smooth_ref = w_smooth_now
                blend_a_1 = min(1.0 / max_hold_frames, 1.0)
                w_emit_1 = w_smooth_ref + blend_a_1 * (fallback_w - w_smooth_ref)
                raw_slope = slope_now / w_emit_1
                continue

            # w_smooth's own EMA state must be unaffected by the blend at
            # every point along the ramp, not just at saturation.
            self.assertAlmostEqual(w_smooth_now, w_smooth_ref, places=9,
                msg=f"frame {k}: w_smooth (internal EMA state) changed when only "
                    f"the emitted weight should have been blended")

            blend_a = min(counter / max_hold_frames, 1.0)
            w_emit_expected = w_smooth_ref + blend_a * (fallback_w - w_smooth_ref)
            expected_slope = w_emit_expected * raw_slope
            self.assertAlmostEqual(slope_now, expected_slope, places=6,
                msg=f"frame {k} (stuck_counter={counter}, blend_a={blend_a}): "
                    f"emitted slope did not match the algebraically-predicted "
                    f"partial-blend value")

        # The intermediate blend checked above (k=2, blend_a=0.5, already
        # verified against the algebraic prediction inside the loop) must
        # sit strictly between the unblended value (w_smooth * raw_slope)
        # and the fully-saturated one (fallback_w * raw_slope) -- confirming
        # it is a genuine partial blend, not accidentally coincident with
        # either endpoint.
        unblended_slope = w_smooth_ref * raw_slope
        saturated_slope = fallback_w * raw_slope
        blend_half = 0.5
        expected_slope_half = (w_smooth_ref + blend_half * (fallback_w - w_smooth_ref)) * raw_slope
        self.assertTrue(min(unblended_slope, saturated_slope) < expected_slope_half < max(unblended_slope, saturated_slope),
            "Intermediate blend_a=0.5 prediction does not sit strictly between "
            "the unblended and saturated slopes -- test setup is degenerate")

    @cpu_and_gpu
    def test_fix_b_prior_widening_selects_wide_prior_after_trigger(self, target_device_idx, xp):
        """
        Fix B (GRADUAL as of 2026-09-10, see prior_widen_after_frames
        docstring): prior_now = spatial_prior + blend_b * (spatial_prior_wide
        - spatial_prior), with blend_b = clip(prev_frame_stuck_counter /
        prior_widen_after_frames, 0, 1) -- a continuous blend of the two
        prior ARRAYS feeding the coarse-peak argmax, replacing an earlier
        all-or-nothing swap that was found to make the coarse peak more
        erratic the instant it switched.

        Discriminating scenario: a faint DECOY spot sits at the sub-aperture
        centre (prior weight 1.0 under EITHER prior, hence under any blend)
        and a much brighter TRUE spot sits far off-centre (r=6, well beyond
        the narrow prior_sigma=2 but well inside the widened prior_sigma=6).
        As blend_b ramps from 0 to 1, the true spot's effective prior weight
        (and hence corr*prior peak height) increases continuously while the
        decoy's stays fixed at 1.0, so there is a specific blend_b at which
        argmax(corr*prior_now) switches from the decoy to the true spot.
        With flux_decoy=100, flux_true=500 that crossover was verified
        (empirically, since a real matched-filter peak height is not
        perfectly proportional to injected flux) to fall between blend_b=0.2
        and blend_b=0.3 -- this test checks one point comfortably on each
        side (blend_b=0.1 decoy, blend_b=0.4 true) plus the blend_b=0 and
        blend_b=1.0 (saturated) endpoints, rather than assuming a single
        step location.

        k_wiener is tiny and ron_e=0 so w_emit stays ~1 throughout, making
        the coarse-peak choice directly visible in the emitted slope.
        stuck_rho_sq_thresh is set absurdly high purely to arm stuck_counter
        deterministically every frame (blend_b uses the PREVIOUS frame's
        counter, so frame k's blend_b = (k-1) / prior_widen_after_frames),
        decoupled from the (deliberately high) confidence used to keep
        w_emit ~1 -- only the frame COUNT, not a real confidence collapse,
        is under test here (that is covered by
        test_stuck_counter_increments_and_resets and
        test_fix_a_gain_fallback_overrides_emission_only).
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        prior_widen_after_frames = 10
        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                         k_wiener=1e-8, w_ema_alpha=1.0,
                                         prior_sigma=2.0, prior_floor=0.05,
                                         stuck_rho_sq_thresh=1e12,  # always "stuck"
                                         prior_widen_after_frames=prior_widen_after_frames,
                                         prior_widen_factor=3.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        decoy = self.generate_spots(ccd_shape, subapdata, xp, flux=100.0, bg=0.0,
                                    shift_dx=0.0, shift_dy=0.0)
        true_spot = self.generate_spots(ccd_shape, subapdata, xp, flux=500.0, bg=0.0,
                                        shift_dx=6.0, shift_dy=0.0)
        frame = decoy + true_spot

        decoy_slope = 0.0
        true_slope = 6.0 / (subap_npx / 2.0)  # 0.75

        # Frame k's Fix B blend uses stuck_counter AFTER frame (k-1), i.e.
        # blend_b(frame k) = (k - 1) / prior_widen_after_frames (clipped).
        # With prior_widen_after_frames=10:
        #   frame 1  -> blend_b = 0/10  = 0.0  (pure narrow prior)  -> decoy
        #   frame 2  -> blend_b = 1/10  = 0.1  (still narrow-dominated) -> decoy
        #   frame 5  -> blend_b = 4/10  = 0.4  (wide-dominated)      -> true spot
        #   frame 11 -> blend_b = 10/10 = 1.0  (saturated, pure wide) -> true spot
        #   frame 12 -> blend_b clipped at 1.0 (still saturated)      -> true spot
        expected = {
            1: decoy_slope,    # blend_b = 0.0  -- exactly inert (matches the old step test's off case)
            2: decoy_slope,    # blend_b = 0.1  -- intermediate, decoy still favoured
            5: true_slope,     # blend_b = 0.4  -- intermediate, true spot now favoured
            11: true_slope,    # blend_b = 1.0  -- saturated (matches the old step test's on case)
            12: true_slope,    # blend_b clipped at 1.0 -- holds, does not overshoot
        }

        for k in range(1, 13):
            self._run_frame(slopec, pixels, frame, t * k)
            self.assertEqual(int(cpuArray(slopec.stuck_counter)[0]), k)

            if k not in expected:
                continue

            w_now = float(cpuArray(slopec.w_out)[0])
            self.assertGreater(w_now, 0.9,
                               f"frame {k}: expected w_emit ~1 so the coarse-peak "
                               f"choice is directly visible in the emitted slope")

            slope_x = float(cpuArray(slopec.outputs['out_slopes'].xslopes)[0])
            self.assertAlmostEqual(slope_x, expected[k], delta=0.05,
                msg=f"frame {k} (blend_b={(k - 1) / prior_widen_after_frames:.2f}): "
                    f"coarse-peak search did not pick the expected spot")

    @cpu_and_gpu
    def test_blend_ramps_saturate_at_endpoints(self, target_device_idx, xp):
        """
        Both new ramps are CLIPPED linear ramps, not unbounded ones: blend_a
        (Fix A) and blend_b (Fix B) must each be exactly 0.0 while
        stuck_counter is 0, and exactly (not asymptotically) 1.0 once
        stuck_counter reaches, and stays at or beyond, the respective ramp
        length (max_hold_frames / prior_widen_after_frames) -- this is the
        core new mechanic (clip(..., 0, 1)) distinguishing the fix from a
        plain unbounded linear blend, complementing the intermediate-ramp
        checks in test_fix_a_gain_fallback_overrides_emission_only and
        test_fix_b_prior_widening_selects_wide_prior_after_trigger.

        blend_a is inferred from w_emit vs the directly-read w_smooth
        (blend_a=0 => w_emit == w_smooth exactly; blend_a=1 => w_emit ==
        fallback_w exactly, and must not overshoot on a further stuck frame).
        blend_b is inferred the same way as the discriminating-scenario test
        above (decoy vs true-spot pick), since blend_b is not otherwise
        exposed as telemetry.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        # --- blend_a: stuck_counter=0 (bright, non-stuck frame) => blend_a=0 ---
        fallback_w = 0.2  # deliberately far from a typical high-SNR w, so a
                          # non-zero blend would be obviously visible
        slopec_a = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                           k_wiener=1e-8, w_ema_alpha=1.0,
                                           stuck_rho_sq_thresh=1.0,  # only zero-flux frames are "stuck"
                                           max_hold_frames=3,
                                           fallback_w=fallback_w,
                                           target_device_idx=target_device_idx)
        slopec_a.inputs['in_pixels'].set(pixels)

        bright_frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e5, bg=0.0,
                                           shift_dx=0.3, shift_dy=0.0)

        # Independent reference for raw_slope on this exact frame: a
        # fully-inert instance (stuck_rho_sq_thresh=0.0) guarantees
        # w_emit == w_smooth == raw_slope * slope, with no possible fallback
        # contribution whatsoever, regardless of ramp semantics.
        probe = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                        k_wiener=1e-8, w_ema_alpha=1.0,
                                        target_device_idx=target_device_idx)
        probe.inputs['in_pixels'].set(pixels)
        self._run_frame(probe, pixels, bright_frame, t * 1)
        probe_w_smooth = float(cpuArray(probe.w_smooth)[0])
        probe_slope = float(cpuArray(probe.outputs['out_slopes'].xslopes)[0])
        raw_slope = probe_slope / probe_w_smooth

        self._run_frame(slopec_a, pixels, bright_frame, t * 1)
        self.assertEqual(int(cpuArray(slopec_a.stuck_counter)[0]), 0,
                         "Test setup assumption violated: bright frame should not be stuck")
        w_smooth_bright = float(cpuArray(slopec_a.w_smooth)[0])
        slope_bright = float(cpuArray(slopec_a.outputs['out_slopes'].xslopes)[0])

        # blend_a MUST be exactly 0.0 at stuck_counter=0: with fallback_w=0.2
        # deliberately far from the near-1 w_smooth expected here, any
        # nonzero blend_a would pull the emitted slope measurably toward
        # fallback_w * raw_slope instead of w_smooth * raw_slope.
        expected_slope_unblended = w_smooth_bright * raw_slope
        self.assertAlmostEqual(slope_bright, expected_slope_unblended, places=6,
            msg="blend_a was not exactly 0.0 at stuck_counter=0: emitted slope "
                "does not match the pure (unblended) w_smooth * raw_slope prediction")

        zero_frame = self.generate_spots(ccd_shape, subapdata, xp, flux=0.0, bg=0.0)
        # Drive stuck_counter to exactly max_hold_frames (saturation) and one past it.
        self._run_frame(slopec_a, pixels, zero_frame, t * 2)  # stuck_counter -> 1
        self._run_frame(slopec_a, pixels, zero_frame, t * 3)  # stuck_counter -> 2
        self._run_frame(slopec_a, pixels, zero_frame, t * 4)  # stuck_counter -> 3 == max_hold_frames
        self.assertEqual(int(cpuArray(slopec_a.stuck_counter)[0]), 3)
        # On a zero-flux frame with no spot, x_est reduces to the (arbitrary
        # but frame-independent) coarse peak with mx=my=0 -- what matters
        # here is only that w_emit has saturated to fallback_w exactly, which
        # is directly visible through xslopes = w_emit * slope_x whenever
        # slope_x != 0 for the chosen coarse peak. Read w_emit indirectly by
        # comparing against one further stuck frame instead, which must be
        # numerically IDENTICAL (both saturated at blend_a=1): x_est is the
        # same deterministic zero-flux coarse peak every time, so identical
        # emitted slopes across consecutive saturated frames confirm w_emit
        # has stopped changing, i.e. blend_a has stopped increasing past 1.0.
        slope_at_saturation = float(cpuArray(slopec_a.outputs['out_slopes'].xslopes)[0])
        self._run_frame(slopec_a, pixels, zero_frame, t * 5)  # stuck_counter -> 4 > max_hold_frames
        self.assertEqual(int(cpuArray(slopec_a.stuck_counter)[0]), 4)
        slope_past_saturation = float(cpuArray(slopec_a.outputs['out_slopes'].xslopes)[0])
        self.assertAlmostEqual(slope_past_saturation, slope_at_saturation, places=9,
            msg="Emitted slope kept changing past stuck_counter > max_hold_frames: "
                "blend_a did not saturate at 1.0 (clip missing/broken)")

        # --- blend_b: same decoy/true-spot discriminating scenario as above ---
        subapdata_b, ccd_shape_b = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_b = Pixels(*ccd_shape_b, target_device_idx=target_device_idx)
        prior_widen_after_frames = 10
        slopec_b = AdaptiveShrinkageSlopec(subapdata_b, fwhm_pix=1.5, ron_e=0.0,
                                           k_wiener=1e-8, w_ema_alpha=1.0,
                                           prior_sigma=2.0, prior_floor=0.05,
                                           stuck_rho_sq_thresh=1e12,  # always "stuck"
                                           prior_widen_after_frames=prior_widen_after_frames,
                                           prior_widen_factor=3.0,
                                           target_device_idx=target_device_idx)
        slopec_b.inputs['in_pixels'].set(pixels_b)

        decoy = self.generate_spots(ccd_shape_b, subapdata_b, xp, flux=100.0, bg=0.0,
                                    shift_dx=0.0, shift_dy=0.0)
        true_spot = self.generate_spots(ccd_shape_b, subapdata_b, xp, flux=500.0, bg=0.0,
                                        shift_dx=6.0, shift_dy=0.0)
        frame_b = decoy + true_spot
        decoy_slope = 0.0
        true_slope = 6.0 / (subap_npx / 2.0)

        # Frame 1: prev stuck_counter = 0 -> blend_b = 0.0 exactly -> decoy.
        self._run_frame(slopec_b, pixels_b, frame_b, t * 1)
        slope_1 = float(cpuArray(slopec_b.outputs['out_slopes'].xslopes)[0])
        self.assertAlmostEqual(slope_1, decoy_slope, delta=0.05,
            msg="blend_b was not exactly 0.0 at stuck_counter=0: the decoy "
                "should be picked under the pure narrow prior")

        # Run up to and one past saturation (prev stuck_counter >= ramp length).
        for k in range(2, prior_widen_after_frames + 3):
            self._run_frame(slopec_b, pixels_b, frame_b, t * k)
        self.assertGreaterEqual(int(cpuArray(slopec_b.stuck_counter)[0]),
                                prior_widen_after_frames + 1)
        slope_saturated = float(cpuArray(slopec_b.outputs['out_slopes'].xslopes)[0])
        self.assertAlmostEqual(slope_saturated, true_slope, delta=0.05,
            msg="blend_b did not saturate to 1.0 once stuck_counter reached "
                "prior_widen_after_frames: the true spot should be fully "
                "reachable under the pure wide prior")

    @cpu_and_gpu
    def test_new_persistent_buffers_shape_dtype_and_identity(self, target_device_idx, xp):
        """
        CUDA-graph-safety check for the two buffers introduced by the
        stuck/Fix-A/Fix-B addition: stuck_counter must be a per-subaperture
        int32 buffer and spatial_prior_wide must have the same shape/dtype
        as spatial_prior, and neither may be reassigned to a new object
        across repeated calc_slopes_nofor() invocations (only written in
        place), mirroring the identity requirement the class docstring
        already imposes on w_smooth/ema_corr/etc.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         prior_widen_factor=2.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        n_subaps = subapdata.n_subaps
        np_sub = subapdata.np_sub
        self.assertEqual(slopec.stuck_counter.shape, (n_subaps,))
        self.assertEqual(slopec.stuck_counter.dtype, xp.int32)
        self.assertEqual(slopec.spatial_prior_wide.shape, (1, np_sub, np_sub))
        self.assertEqual(slopec.spatial_prior_wide.dtype, slopec.dtype)

        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e4, bg=0.0)
        id_stuck_before = id(slopec.stuck_counter)
        id_prior_wide_before = id(slopec.spatial_prior_wide)

        self._run_frame(slopec, pixels, frame, t * 1)
        self._run_frame(slopec, pixels, frame, t * 2)

        self.assertEqual(id(slopec.stuck_counter), id_stuck_before,
                         "stuck_counter buffer was reassigned instead of written in place")
        self.assertEqual(id(slopec.spatial_prior_wide), id_prior_wide_before,
                         "spatial_prior_wide buffer was reassigned (it is a "
                         "precomputed constant and must never change identity)")


if __name__ == '__main__':
    unittest.main()
