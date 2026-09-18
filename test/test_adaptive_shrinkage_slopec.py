import specula
specula.init(0)  # Default target device

import inspect
import unittest

from specula import np
from specula import cpuArray
from specula.base_value import BaseValue

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

    @cpu_and_gpu
    def test_halo_fraction_zero_matches_single_gaussian_template_exactly(self, target_device_idx, xp):
        """
        Backward compatibility for the 2026-09-13 core+halo matched-filter
        template (see halo_fwhm_pix/halo_fraction docstrings): the default
        halo_fraction=0.0 must leave fft_template_conj BIT-IDENTICAL to the
        old single-Gaussian template, both when halo_fwhm_pix/halo_fraction
        are simply not passed and when halo_fwhm_pix is explicitly set to
        some value but halo_fraction is left at 0.0 (halo_fraction, not
        halo_fwhm_pix alone, must gate whether the halo contributes at all).
        """
        subap_npx = 16
        subapdata, _ = self.get_test_setup(target_device_idx, xp, subap_npx)

        s_no_halo_kwargs = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5,
                                                   target_device_idx=target_device_idx)
        s_explicit_zero_fraction = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5,
                                                            halo_fwhm_pix=6.0, halo_fraction=0.0,
                                                            target_device_idx=target_device_idx)

        np.testing.assert_array_equal(cpuArray(s_explicit_zero_fraction.fft_template_conj),
            cpuArray(s_no_halo_kwargs.fft_template_conj),
            err_msg="halo_fraction=0.0 with halo_fwhm_pix set must reproduce the "
                    "single-Gaussian template bit-for-bit")

    @cpu_and_gpu
    def test_halo_component_adds_measurable_far_wing_flux(self, target_device_idx, xp):
        """
        With halo_fwhm_pix set and halo_fraction > 0, the real-space template
        (recovered from fft_template_conj) must carry measurably more flux
        far from its centre than the pure-core (halo_fraction=0) template at
        the same fwhm_pix -- confirms the halo Gaussian genuinely contributes
        mass to the matched filter, not just a no-op rescaling of the core.
        """
        subap_npx = 16
        subapdata, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        np_sub = subapdata.np_sub

        slopec_core = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, halo_fwhm_pix=6.0,
                                              halo_fraction=0.0, target_device_idx=target_device_idx)
        slopec_mix = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, halo_fwhm_pix=6.0,
                                             halo_fraction=0.3, target_device_idx=target_device_idx)

        def real_space_template(slopec):
            # fft_template_conj = conj(FFT(template)) -> template = IFFT(conj(fft_template_conj)).
            conj_fft = xp.conj(slopec.fft_template_conj)
            return xp.fft.ifft2(conj_fft, axes=(1, 2)).real[0]

        t_core = cpuArray(real_space_template(slopec_core))
        t_mix = cpuArray(real_space_template(slopec_mix))

        # Centre the FFT-origin templates so a simple Euclidean radius from
        # the array centre is meaningful (matches the class's own
        # FFT-origin convention, just re-centred for measurement here).
        t_core_centred = np.fft.fftshift(t_core)
        t_mix_centred = np.fft.fftshift(t_mix)

        cy = cx = np_sub // 2
        yy, xx = np.mgrid[0:np_sub, 0:np_sub]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        far_wing = r > 4.0

        wing_flux_core = float(np.sum(t_core_centred[far_wing]))
        wing_flux_mix = float(np.sum(t_mix_centred[far_wing]))

        self.assertGreater(wing_flux_mix, 10.0 * wing_flux_core,
            f"Two-component template did not show measurably more far-wing "
            f"flux than the pure-core template (core={wing_flux_core}, "
            f"mix={wing_flux_mix})")

    @cpu_and_gpu
    def test_two_component_template_stays_normalized(self, target_device_idx, xp):
        """
        Both core and halo are individually normalized to sum 1 before being
        mixed with weights (1 - halo_fraction) and halo_fraction (which
        themselves sum to 1), so the resulting real-space template must still
        integrate to ~1, exactly like the original single-Gaussian template.
        """
        subap_npx = 16
        subapdata, _ = self.get_test_setup(target_device_idx, xp, subap_npx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, halo_fwhm_pix=6.0,
                                         halo_fraction=0.3, target_device_idx=target_device_idx)

        conj_fft = xp.conj(slopec.fft_template_conj)
        template = cpuArray(xp.fft.ifft2(conj_fft, axes=(1, 2)).real[0])

        self.assertAlmostEqual(float(np.sum(template)), 1.0, places=5,
            msg="Two-component matched-filter template does not integrate to 1")

    @cpu_and_gpu
    def test_halo_fraction_one_reduces_to_pure_halo_gaussian(self, target_device_idx, xp):
        """
        Boundary case: halo_fraction=1.0 must give the core component zero
        weight, so fft_template_conj should be bit-identical to the template
        of an instance built with fwhm_pix set directly to halo_fwhm_pix's
        value (no halo at all) -- i.e. a pure single Gaussian at the halo's
        width, with no residual core contribution.
        """
        subap_npx = 16
        subapdata, _ = self.get_test_setup(target_device_idx, xp, subap_npx)

        slopec_frac_one = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, halo_fwhm_pix=6.0,
                                                  halo_fraction=1.0, target_device_idx=target_device_idx)
        slopec_pure_halo_width = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=6.0,
                                                          target_device_idx=target_device_idx)

        np.testing.assert_array_equal(cpuArray(slopec_frac_one.fft_template_conj),
            cpuArray(slopec_pure_halo_width.fft_template_conj),
            err_msg="halo_fraction=1.0 did not reduce cleanly to a pure Gaussian "
                    "at halo_fwhm_pix's width (core contribution not fully zeroed)")

    @cpu_and_gpu
    def test_two_component_template_localizes_core_halo_spot(self, target_device_idx, xp):
        """
        Sanity check that coarse-peak localization still works with a
        two-component template on a synthetic spot that itself has a
        core+halo structure (built by summing two calls to the existing
        generate_spots() helper at the core and halo FWHMs and a shared
        sub-pixel shift, rather than reinventing spot generation). With
        shrinkage neutralized (as in
        test_subpixel_accuracy_when_shrinkage_neutralized), the emitted
        slope -- read via calc_slopes_nofor()'s own output, not a raw
        argmax -- must track the injected shift in sign and rough magnitude.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        core_fwhm, halo_fwhm = 1.5, 6.0
        shift_x, shift_y = 0.3, -0.2
        core = self.generate_spots(ccd_shape, subapdata, xp, fwhm=core_fwhm, flux=700.0,
                                   shift_dx=shift_x, shift_dy=shift_y)
        halo = self.generate_spots(ccd_shape, subapdata, xp, fwhm=halo_fwhm, flux=300.0,
                                   shift_dx=shift_x, shift_dy=shift_y)
        frame = core + halo

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=core_fwhm, halo_fwhm_pix=halo_fwhm,
                                         halo_fraction=0.3, k_wiener=1e-8, b_reg=0.0, ron_e=0.0,
                                         w_ema_alpha=1.0, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        self._run_frame(slopec, pixels, frame, t)

        w_out = float(cpuArray(slopec.w_out)[0])
        self.assertGreater(w_out, 0.99,
                           "w_out did not approach 1 with shrinkage neutralized")

        slopes_x = cpuArray(slopec.outputs['out_slopes'].xslopes)
        slopes_y = cpuArray(slopec.outputs['out_slopes'].yslopes)
        expected_slope_x = shift_x / (subap_npx / 2.0)
        expected_slope_y = shift_y / (subap_npx / 2.0)

        np.testing.assert_allclose(slopes_x, expected_slope_x, atol=0.05,
                                   err_msg="X localization on a core+halo spot failed with "
                                           "a two-component template")
        np.testing.assert_allclose(slopes_y, expected_slope_y, atol=0.05,
                                   err_msg="Y localization on a core+halo spot failed with "
                                           "a two-component template")

    @cpu_and_gpu
    def test_x_c_y_c_outputs_exist_with_correct_type_and_shape(self, target_device_idx, xp):
        """
        out_x_c/out_y_c (2026-09-14, see the class __init__ comment on
        Step-1 coarse-peak telemetry) must be registered in self.outputs as
        BaseValue instances, each holding one value per sub-aperture -- same
        registration pattern as out_w_smooth/out_gamma/out_rho_sq. Uses a
        2x2 sub-aperture grid (n_subaps=4), not the usual single-subap
        default, so the shape check is not trivially satisfied by n_subaps
        happening to be 1.
        """
        subap_npx = 8
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx, n_sub_side=2)
        n_subaps = subapdata.n_subaps
        self.assertEqual(n_subaps, 4)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         target_device_idx=target_device_idx)

        for name in ('out_x_c', 'out_y_c'):
            self.assertIn(name, slopec.outputs, f"{name} missing from self.outputs")
            self.assertIsInstance(slopec.outputs[name], BaseValue,
                                  f"{name} is not a BaseValue instance")
            self.assertEqual(slopec.outputs[name].value.shape, (n_subaps,),
                             f"{name} does not have one value per sub-aperture")

    @cpu_and_gpu
    def test_x_c_y_c_populated_by_trigger_code(self, target_device_idx, xp):
        """
        out_x_c/out_y_c must be written by trigger_code(), not left at their
        zero-initialized __init__ default. A centred, high-flux spot on a
        16x16 (even-sized, offset=0.5) sub-aperture has its coarse peak
        exactly at the array centre pixel, i.e. x_c == y_c == cntrd == 7.5
        -- a specific, predictable value, not just "non-zero" (which would
        also pass by accident if the buffer were left uninitialized garbage
        near zero for the wrong reason).
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        cntrd = (subap_npx - 1) / 2.0

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Before any frame: still at the __init__ zero default.
        x_c_before = cpuArray(slopec.outputs['out_x_c'].value)
        y_c_before = cpuArray(slopec.outputs['out_y_c'].value)
        np.testing.assert_array_equal(x_c_before, np.zeros_like(x_c_before))
        np.testing.assert_array_equal(y_c_before, np.zeros_like(y_c_before))

        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e5, bg=0.0)
        self._run_frame(slopec, pixels, frame, t)

        x_c_after = cpuArray(slopec.outputs['out_x_c'].value)
        y_c_after = cpuArray(slopec.outputs['out_y_c'].value)
        np.testing.assert_allclose(x_c_after, cntrd, atol=1e-9,
            err_msg="out_x_c was not updated to the expected centred coarse peak")
        np.testing.assert_allclose(y_c_after, cntrd, atol=1e-9,
            err_msg="out_y_c was not updated to the expected centred coarse peak")

        # Also mirrored on the internal generation_time, like the other
        # Effective-gain telemetry outputs (see post_trigger()).
        self.assertEqual(slopec.outputs['out_x_c'].generation_time, t)
        self.assertEqual(slopec.outputs['out_y_c'].generation_time, t)

    @cpu_and_gpu
    def test_x_c_y_c_are_always_integer_plus_offset(self, target_device_idx, xp):
        """
        x_c/y_c are computed as `x_idx.astype(dtype) + self.offset` straight
        from an `xp.argmax` over the correlation map (Step 1) -- they must
        therefore always land at an integer-plus-offset grid position,
        NEVER a genuinely fractional/interpolated value (sub-pixel
        refinement only happens in Step 2/3, downstream of x_c/y_c). Checked
        over a range of injected sub-pixel shifts and BOTH sub-aperture
        parities, since `offset` itself depends on `np_sub % 2`
        (0.5 for even, 0.0 for odd).
        """
        t = int(1e9)
        for subap_npx in (15, 16):  # odd (offset=0.0) and even (offset=0.5)
            subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
            pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
            expected_offset = 0.5 if subap_npx % 2 == 0 else 0.0

            slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                             target_device_idx=target_device_idx)
            slopec.inputs['in_pixels'].set(pixels)
            self.assertEqual(slopec.offset, expected_offset)

            for i, (shift_dx, shift_dy) in enumerate(
                    [(0.0, 0.0), (0.3, -0.2), (-0.45, 0.45), (0.49, -0.49)], start=1):
                frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e4, bg=0.0,
                                            shift_dx=shift_dx, shift_dy=shift_dy)
                self._run_frame(slopec, pixels, frame, t * i)

                x_c = cpuArray(slopec.outputs['out_x_c'].value)
                y_c = cpuArray(slopec.outputs['out_y_c'].value)

                x_frac = x_c - expected_offset
                y_frac = y_c - expected_offset
                np.testing.assert_allclose(x_frac, np.round(x_frac), atol=1e-6,
                    err_msg=f"np_sub={subap_npx}, shift=({shift_dx},{shift_dy}): "
                            f"x_c - offset is not integer-valued: {x_frac}")
                np.testing.assert_allclose(y_frac, np.round(y_frac), atol=1e-6,
                    err_msg=f"np_sub={subap_npx}, shift=({shift_dx},{shift_dy}): "
                            f"y_c - offset is not integer-valued: {y_frac}")

    @cpu_and_gpu
    def test_x_c_y_c_telemetry_has_no_effect_on_emitted_slopes(self, target_device_idx, xp):
        """
        out_x_c/out_y_c are documented as TELEMETRY ONLY (see the __init__
        comment introducing them). This is checked two ways rather than
        assumed:

        1. Source-order inspection: self.x_c_out/self.y_c_out/x_c_value/
           y_c_value are only ever written (`self.x_c_out[:] = x_c`, etc.),
           and that write happens AFTER `self.slopes.xslopes`/`yslopes` are
           already assigned from x_est/y_est/w_emit -- i.e. textually,
           inside calc_slopes_nofor(), none of these four names appear
           before the slopes are set, so they cannot feed back into the
           slopes computation even in principle (this would catch a future
           refactor that accidentally started reading them back in).
        2. Behavioural: two fresh, identically-configured instances run the
           exact same varying (flux/shift/noise) frame sequence; one never
           has its out_x_c/out_y_c outputs read at all during the run, the
           other has them read after every single frame. Reading (or not
           reading) a telemetry-only output must not change out_slopes --
           the two instances' emitted slopes must match bit-for-bit.
        """
        source = inspect.getsource(AdaptiveShrinkageSlopec.calc_slopes_nofor)
        marker = "self.slopes.xslopes = w_emit * slope_x"
        self.assertIn(marker, source, "calc_slopes_nofor() source changed shape; "
                                      "update this test's marker line")
        before_slopes, _, after_slopes = source.partition(marker)
        for name in ('x_c_out', 'y_c_out', 'x_c_value', 'y_c_value'):
            self.assertNotIn(name, before_slopes,
                f"'{name}' is referenced before out_slopes is assigned in "
                f"calc_slopes_nofor() -- it may no longer be telemetry-only")
            self.assertIn(name, after_slopes,
                f"'{name}' is never written after out_slopes is assigned -- "
                f"expected the Step-1 coarse-peak telemetry write here")

        subap_npx, t = 16, int(1e9)
        subapdata_a, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_b, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_a = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_b = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        kwargs = dict(fwhm_pix=1.5, k_wiener=10.0, b_reg=0.5, ron_e=1.0,
                     w_ema_alpha=0.2, target_device_idx=target_device_idx)
        slopec_unread = AdaptiveShrinkageSlopec(subapdata_a, **kwargs)
        slopec_read = AdaptiveShrinkageSlopec(subapdata_b, **kwargs)
        slopec_unread.inputs['in_pixels'].set(pixels_a)
        slopec_read.inputs['in_pixels'].set(pixels_b)

        rng = np.random.RandomState(99)
        for i in range(1, 31):
            flux = float(rng.choice([0.0, 1.0, 10.0, 1e3, 1e5]))
            shift_dx = float(rng.uniform(-0.4, 0.4))
            shift_dy = float(rng.uniform(-0.4, 0.4))
            noise_std = float(rng.uniform(0.0, 1.5))
            frame = self.generate_spots(ccd_shape, subapdata_a, xp, flux=flux, bg=0.5,
                                        shift_dx=shift_dx, shift_dy=shift_dy,
                                        noise_std=noise_std)
            self._run_frame(slopec_unread, pixels_a, frame, t * i)
            self._run_frame(slopec_read, pixels_b, frame, t * i)

            # Only slopec_read's telemetry is ever touched mid-run.
            _ = cpuArray(slopec_read.outputs['out_x_c'].value)
            _ = cpuArray(slopec_read.outputs['out_y_c'].value)

            xu = cpuArray(slopec_unread.outputs['out_slopes'].xslopes)
            xr = cpuArray(slopec_read.outputs['out_slopes'].xslopes)
            yu = cpuArray(slopec_unread.outputs['out_slopes'].yslopes)
            yr = cpuArray(slopec_read.outputs['out_slopes'].yslopes)

            np.testing.assert_array_equal(xr, xu,
                err_msg=f"frame {i}: xslopes differ depending on whether "
                        f"out_x_c/out_y_c telemetry was read -- not telemetry-only")
            np.testing.assert_array_equal(yr, yu,
                err_msg=f"frame {i}: yslopes differ depending on whether "
                        f"out_x_c/out_y_c telemetry was read -- not telemetry-only")

    @cpu_and_gpu
    def test_x_c_jitters_across_a_pixel_boundary_but_not_within_a_pixel(self, target_device_idx, xp):
        """
        Investigates whether Step 1's un-refined integer-pixel argmax
        causes real frame-to-frame quantization jitter (a candidate
        contributor to ASHR's known window-size sensitivity): a true spot
        sitting near the boundary between two pixels (shift_dx close to
        0.5, i.e. equidistant from the pixel at x_c=7.5 and the one at
        x_c=8.5 on this 16px, offset=0.5 grid) should have its coarse peak
        argmax flip between the two neighbouring pixels from one noisy
        frame to the next, driven by nothing but the noise realization --
        whereas a spot placed solidly within one pixel (shift_dx=0.0, at
        the pixel centre) should keep the SAME coarse peak across the same
        noise realizations. Demonstrating the flip near the boundary (not
        just "the plumbing works") is the actual diagnostic value here.

        Low flux + moderate read noise is used deliberately so the two
        candidate peaks are close enough in matched-filter response for
        noise to plausibly decide between them; this is not tuned to
        reproduce any specific real closed-loop SNR, only to exhibit the
        phenomenon.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        cntrd = (subap_npx - 1) / 2.0

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        n_trials = 40

        def sample_x_c(shift_dx, seed_offset):
            values = []
            for k in range(n_trials):
                np.random.seed(seed_offset + k)
                frame = self.generate_spots(ccd_shape, subapdata, xp, flux=30.0, bg=1.0,
                                            shift_dx=shift_dx, shift_dy=0.0,
                                            noise_std=1.5)
                self._run_frame(slopec, pixels, frame, t * (seed_offset + k + 1))
                values.append(float(cpuArray(slopec.outputs['out_x_c'].value)[0]))
            return values

        # Several similar-but-slightly-different placements straddling the
        # boundary between the pixel at cntrd (7.5) and its neighbour
        # (8.5): each must show BOTH values across repeated noisy frames.
        for shift_dx in (0.45, 0.5, 0.55):
            values = sample_x_c(shift_dx, seed_offset=int(shift_dx * 1000))
            distinct = set(values)
            self.assertEqual(distinct, {cntrd, cntrd + 1.0},
                f"shift_dx={shift_dx}: expected the coarse peak to jitter "
                f"between {cntrd} and {cntrd + 1.0} across noise "
                f"realizations at a pixel boundary, got {sorted(distinct)}")

        # Control: a spot solidly within one pixel (no shift, dead centre)
        # must NOT jitter under the exact same noise realizations/flux.
        values_centred = sample_x_c(0.0, seed_offset=99000)
        self.assertEqual(set(values_centred), {cntrd},
            f"shift_dx=0.0 (pixel centre, well away from any boundary) "
            f"unexpectedly jittered: {sorted(set(values_centred))} -- "
            f"the control case for the boundary-jitter demonstration failed")


    @cpu_and_gpu
    def test_subpixel_peak_refine_default_false_is_pure_no_op(self, target_device_idx, xp):
        """
        subpixel_peak_refine (2026-09-14, see class docstring) defaults to
        False and must then be bit-for-bit identical to a reference instance
        built WITHOUT the parameter at all -- both in the emitted out_slopes
        and in the out_x_c/out_y_c telemetry the feature is designed to
        change when enabled. Also checks subpixel_peak_refine=False passed
        explicitly, not just the unspecified default, against the same
        reference. Uses the same randomized varying-frame-sequence pattern
        as the other "new knob is inert" regression tests in this file
        (e.g. test_stuck_detector_and_both_fixes_default_inert).
        """
        subap_npx, t = 16, int(1e9)
        subapdata_ref, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_false, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_ref = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_false = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        common = dict(fwhm_pix=1.5, k_wiener=10.0, b_reg=0.5, ron_e=1.0,
                     w_ema_alpha=0.2, target_device_idx=target_device_idx)
        slopec_reference = AdaptiveShrinkageSlopec(subapdata_ref, **common)
        slopec_explicit_false = AdaptiveShrinkageSlopec(subapdata_false,
                                                        subpixel_peak_refine=False, **common)
        slopec_reference.inputs['in_pixels'].set(pixels_ref)
        slopec_explicit_false.inputs['in_pixels'].set(pixels_false)

        self.assertFalse(slopec_reference.subpixel_peak_refine,
                         "Default must be False (preserve original behaviour)")

        rng = np.random.RandomState(2026)
        for i in range(1, 41):
            flux = float(rng.choice([0.0, 1.0, 10.0, 1e3, 1e5]))
            shift_dx = float(rng.uniform(-0.4, 0.4))
            shift_dy = float(rng.uniform(-0.4, 0.4))
            noise_std = float(rng.uniform(0.0, 1.5))
            frame = self.generate_spots(ccd_shape, subapdata_ref, xp, flux=flux, bg=0.5,
                                        shift_dx=shift_dx, shift_dy=shift_dy,
                                        noise_std=noise_std)
            self._run_frame(slopec_reference, pixels_ref, frame, t * i)
            self._run_frame(slopec_explicit_false, pixels_false, frame, t * i)

            xr = cpuArray(slopec_reference.outputs['out_slopes'].xslopes)
            xf = cpuArray(slopec_explicit_false.outputs['out_slopes'].xslopes)
            yr = cpuArray(slopec_reference.outputs['out_slopes'].yslopes)
            yf = cpuArray(slopec_explicit_false.outputs['out_slopes'].yslopes)
            xcr = cpuArray(slopec_reference.outputs['out_x_c'].value)
            xcf = cpuArray(slopec_explicit_false.outputs['out_x_c'].value)
            ycr = cpuArray(slopec_reference.outputs['out_y_c'].value)
            ycf = cpuArray(slopec_explicit_false.outputs['out_y_c'].value)

            np.testing.assert_array_equal(xf, xr,
                err_msg=f"frame {i}: xslopes differ with subpixel_peak_refine=False "
                        f"vs the no-parameter reference -- not a true no-op")
            np.testing.assert_array_equal(yf, yr,
                err_msg=f"frame {i}: yslopes differ with subpixel_peak_refine=False "
                        f"vs the no-parameter reference -- not a true no-op")
            np.testing.assert_array_equal(xcf, xcr,
                err_msg=f"frame {i}: out_x_c differs with subpixel_peak_refine=False "
                        f"vs the no-parameter reference -- not a true no-op")
            np.testing.assert_array_equal(ycf, ycr,
                err_msg=f"frame {i}: out_y_c differs with subpixel_peak_refine=False "
                        f"vs the no-parameter reference -- not a true no-op")

    @cpu_and_gpu
    def test_subpixel_peak_refine_improves_small_offset_accuracy(self, target_device_idx, xp):
        """
        Correctness property (not just "outputs exist"): for small, known
        sub-pixel true offsets -- well within a 3-point parabolic fit's
        validity range -- subpixel_peak_refine=True must move out_x_c/
        out_y_c strictly closer to the TRUE injected offset than the raw
        integer-pixel argmax, independently in x and y. High flux, no
        noise, ron_e=0 isolates Step 1's coarse-peak search itself: x_c/y_c
        do not depend on k_wiener/w_ema_alpha/b_reg at all (those only
        affect Step 2/3 and the shrinkage gain further downstream), so no
        shrinkage-neutralizing kwargs are needed here.

        Shifts are applied one axis at a time (the other held at 0) so a
        sign/axis bug in only one of x or y cannot hide behind the other
        axis being correct, mirroring the offset convention already used by
        test_subpixel_accuracy_when_shrinkage_neutralized: true position
        offset from centre == out_x_c - cntrd (== shift_dx by
        generate_spots()'s own convention).

        Confidence gate (2026-09-14, see class docstring's subpixel_peak_refine
        entry): the correction is scaled by w_smooth AS IT STANDS AT THE
        START OF THE FRAME, which is exactly 0 for a freshly-constructed
        instance -- a single frame would therefore see essentially zero
        correction regardless of the true offset. Each scenario below warms
        up on the SAME frame repeated (a genuinely tracked, stable target,
        not a contrived state) so w_smooth's own EMA ramps to a realistic
        high-confidence value (verified directly, not assumed) before the
        frame that is actually checked -- this keeps the original
        correctness property (refined estimate closer to truth than raw)
        meaningful once confidence is actually high, rather than trivially
        satisfied by the gate suppressing everything to ~0.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        cntrd = (subap_npx - 1) / 2.0
        offsets = [-0.3, -0.15, 0.0, 0.15, 0.3]
        n_warmup = 40

        def raw_and_refined_offset(shift_dx, shift_dy, axis):
            pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
            slopec_raw = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                                 target_device_idx=target_device_idx,
                                                 subpixel_peak_refine=False)
            slopec_ref = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                                 target_device_idx=target_device_idx,
                                                 subpixel_peak_refine=True)
            slopec_raw.inputs['in_pixels'].set(pixels)
            slopec_ref.inputs['in_pixels'].set(pixels)
            frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                        shift_dx=shift_dx, shift_dy=shift_dy)
            for i in range(1, n_warmup + 1):
                self._run_frame(slopec_raw, pixels, frame, t * i)
                self._run_frame(slopec_ref, pixels, frame, t * i)

            w_smooth_ref = float(cpuArray(slopec_ref.w_smooth)[0])
            self.assertGreater(w_smooth_ref, 0.999,
                f"axis={axis}, true_offset={shift_dx or shift_dy}: test setup "
                f"assumption violated -- expected w_smooth to have warmed up "
                f"close to 1 after {n_warmup} identical high-SNR frames")

            out_name = 'out_x_c' if axis == 'x' else 'out_y_c'
            raw_val = float(cpuArray(slopec_raw.outputs[out_name].value)[0]) - cntrd
            ref_val = float(cpuArray(slopec_ref.outputs[out_name].value)[0]) - cntrd
            return raw_val, ref_val

        for axis, true_offset in [(a, o) for a in ('x', 'y') for o in offsets]:
            shift_dx = true_offset if axis == 'x' else 0.0
            shift_dy = true_offset if axis == 'y' else 0.0
            raw_val, ref_val = raw_and_refined_offset(shift_dx, shift_dy, axis)

            if true_offset == 0.0:
                self.assertAlmostEqual(ref_val, 0.0, places=9,
                    msg=f"axis={axis}: refined offset should be exactly 0 for a "
                        f"dead-centre spot, got {ref_val}")
                continue

            raw_err = abs(raw_val - true_offset)
            ref_err = abs(ref_val - true_offset)
            self.assertLess(ref_err, raw_err,
                f"axis={axis}, true_offset={true_offset}: refined estimate "
                f"({ref_val}) is not closer to the true offset than the raw "
                f"integer-pixel one ({raw_val})")
            self.assertLess(ref_err, 0.05,
                f"axis={axis}, true_offset={true_offset}: refined estimate "
                f"({ref_val}) is not within 0.05px of the true offset")

    @cpu_and_gpu
    def test_subpixel_peak_refine_no_correction_for_dead_centre_spot(self, target_device_idx, xp):
        """
        A spot placed dead-centre on a pixel (shift_dx=shift_dy=0, where the
        raw integer argmax already lands exactly on the true centre, see
        test_x_c_y_c_populated_by_trigger_code) must get zero (or numerically
        negligible) correction from refinement: the 3-point parabolic fit is
        symmetric about its own centre sample, so f(-1) == f(+1) exactly in
        the noiseless case and dx/dy must be exactly 0.

        Confidence gate (2026-09-14): at w_smooth=0 (fresh instance / frame
        1) ANY dx the parabolic fit produced would be suppressed to 0 by the
        gate itself, which would make this check pass for the wrong reason
        (masking the fit's own symmetry, not exercising it). Warms up on the
        same dead-centre frame first so w_smooth is genuinely high
        (confirmed directly) before the assertion frame.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        cntrd = (subap_npx - 1) / 2.0

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                         subpixel_peak_refine=True,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0)
        for i in range(1, 41):
            self._run_frame(slopec, pixels, frame, t * i)

        w_smooth = float(cpuArray(slopec.w_smooth)[0])
        self.assertGreater(w_smooth, 0.999,
            "Test setup assumption violated: expected w_smooth to have "
            "warmed up close to 1 after 40 identical high-SNR frames")

        x_c = float(cpuArray(slopec.outputs['out_x_c'].value)[0])
        y_c = float(cpuArray(slopec.outputs['out_y_c'].value)[0])
        self.assertAlmostEqual(x_c, cntrd, places=9,
            msg="Refinement injected a spurious x correction for a dead-centre spot")
        self.assertAlmostEqual(y_c, cntrd, places=9,
            msg="Refinement injected a spurious y correction for a dead-centre spot")

    @cpu_and_gpu
    def test_subpixel_peak_refine_flatness_fallback_on_zero_flux(self, target_device_idx, xp):
        """
        Degenerate-curvature fallback: on a literal all-zero (no spot, no
        background) frame the prior-weighted correlation map self._tmp is
        identically zero everywhere, so the peak, both its neighbours and
        the curvature (den_x/den_y) are all exactly 0 -- the flatness guard
        (|den| < min_curvature) must trigger and fall back to dx=dy=0,
        rather than the 0/0 curvature ratio producing NaN/Inf or a wild
        clipped +-0.5 value. Checked by comparing directly against a
        subpixel_peak_refine=False instance fed the exact same frame: with
        the correction falling back to exactly 0, out_x_c/out_y_c must be
        numerically identical between the two, not merely "some finite
        number".

        Confidence gate (2026-09-14): both instances are first warmed up on
        a bright, well-tracked spot so w_smooth is genuinely high going into
        the all-zero assertion frame -- at w_smooth=0 (fresh instance /
        frame 1) the flatness fallback would be trivially masked by the
        gate itself rather than actually exercised.
        """
        subap_npx, t = 16, int(1e9)
        subapdata_off, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_on, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_off = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_on = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec_off = AdaptiveShrinkageSlopec(subapdata_off, fwhm_pix=1.5, ron_e=0.0,
                                             subpixel_peak_refine=False,
                                             target_device_idx=target_device_idx)
        slopec_on = AdaptiveShrinkageSlopec(subapdata_on, fwhm_pix=1.5, ron_e=0.0,
                                            subpixel_peak_refine=True,
                                            target_device_idx=target_device_idx)
        slopec_off.inputs['in_pixels'].set(pixels_off)
        slopec_on.inputs['in_pixels'].set(pixels_on)

        bright_frame = self.generate_spots(ccd_shape, subapdata_on, xp, flux=1e6, bg=0.0)
        for i in range(1, 41):
            self._run_frame(slopec_off, pixels_off, bright_frame, t * i)
            self._run_frame(slopec_on, pixels_on, bright_frame, t * i)

        w_smooth_on = float(cpuArray(slopec_on.w_smooth)[0])
        self.assertGreater(w_smooth_on, 0.999,
            "Test setup assumption violated: expected w_smooth to have "
            "warmed up close to 1 before the all-zero assertion frame")

        zero_frame = xp.zeros(ccd_shape, dtype=xp.float32)
        self._run_frame(slopec_off, pixels_off, zero_frame, t * 41)
        self._run_frame(slopec_on, pixels_on, zero_frame, t * 41)

        x_c_off = cpuArray(slopec_off.outputs['out_x_c'].value)
        y_c_off = cpuArray(slopec_off.outputs['out_y_c'].value)
        x_c_on = cpuArray(slopec_on.outputs['out_x_c'].value)
        y_c_on = cpuArray(slopec_on.outputs['out_y_c'].value)

        self.assertTrue(np.all(np.isfinite(x_c_on)), "out_x_c is not finite on an all-zero frame")
        self.assertTrue(np.all(np.isfinite(y_c_on)), "out_y_c is not finite on an all-zero frame")
        np.testing.assert_array_equal(x_c_on, x_c_off,
            err_msg="Flatness fallback did not produce dx=0: out_x_c differs from "
                    "the unrefined value on a perfectly flat (all-zero) correlation map")
        np.testing.assert_array_equal(y_c_on, y_c_off,
            err_msg="Flatness fallback did not produce dy=0: out_y_c differs from "
                    "the unrefined value on a perfectly flat (all-zero) correlation map")

    @cpu_and_gpu
    def test_subpixel_peak_refine_wraps_periodically_at_subaperture_edge(self, target_device_idx, xp):
        """
        Neighbour lookup wraps periodically (`% np_sub`), matching the
        correlation's own FFT-circular topology (see class docstring) --
        checked at BOTH ends of the sub-aperture, where the coarse peak's
        integer argmax sits at index 0 (neighbour -1 must wrap to
        np_sub - 1) or at index np_sub - 1 (neighbour +1 must wrap to 0).
        A widened, near-flat spatial prior (large prior_sigma, prior_floor
        near 1) lets a spot placed near the array edge still win the
        coarse-peak search, instead of the default prior suppressing it.
        Only requires no crash and a finite, bounded (within +-0.5 px of
        the raw integer peak) result -- not a specific numeric value, since
        the true spot centre lies right at (or past) the sampled edge,
        outside a 3-point fit's normal validity range.

        Confidence gate (2026-09-14): both instances are warmed up on the
        same edge-placed frame before the assertion so w_smooth is
        genuinely high -- otherwise (fresh instance / frame 1, w_smooth=0)
        the bound check below would be trivially satisfied by the gate
        suppressing the correction to ~0, not by the wraparound logic
        actually being exercised. A minimum-magnitude check confirms the
        correction is meaningfully non-zero once confidence is high.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        cntrd = (subap_npx - 1) / 2.0

        wide_prior_kwargs = dict(fwhm_pix=1.5, ron_e=0.0, prior_sigma=1000.0, prior_floor=1.0,
                                 target_device_idx=target_device_idx)

        for shift_dx, edge_idx, label in [(-cntrd, 0, "left"), (cntrd, subap_npx - 1, "right")]:
            slopec_raw = AdaptiveShrinkageSlopec(subapdata, subpixel_peak_refine=False, **wide_prior_kwargs)
            slopec_ref = AdaptiveShrinkageSlopec(subapdata, subpixel_peak_refine=True, **wide_prior_kwargs)
            slopec_raw.inputs['in_pixels'].set(pixels)
            slopec_ref.inputs['in_pixels'].set(pixels)

            frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                        shift_dx=shift_dx, shift_dy=0.0)
            try:
                for i in range(1, 41):
                    self._run_frame(slopec_raw, pixels, frame, t * i)
                    self._run_frame(slopec_ref, pixels, frame, t * i)
            except Exception as e:  # pragma: no cover - failure path
                self.fail(f"{label} edge: refinement raised at the sub-aperture "
                          f"boundary (wraparound bug?): {e!r}")

            w_smooth_ref = float(cpuArray(slopec_ref.w_smooth)[0])
            self.assertGreater(w_smooth_ref, 0.999,
                f"{label} edge: test setup assumption violated -- expected "
                f"w_smooth to have warmed up close to 1 after 40 identical "
                f"high-SNR frames")

            x_c_raw = float(cpuArray(slopec_raw.outputs['out_x_c'].value)[0])
            x_c_ref = float(cpuArray(slopec_ref.outputs['out_x_c'].value)[0])
            y_c_ref = float(cpuArray(slopec_ref.outputs['out_y_c'].value)[0])

            self.assertTrue(np.isfinite(x_c_ref), f"{label} edge: out_x_c is not finite")
            self.assertTrue(np.isfinite(y_c_ref), f"{label} edge: out_y_c is not finite")
            self.assertLessEqual(abs(x_c_ref - x_c_raw), 0.5 + 1e-9,
                f"{label} edge: refined x_c ({x_c_ref}) strayed more than the "
                f"clip(-0.5, 0.5) safety bound from the raw integer peak ({x_c_raw})")
            self.assertGreater(abs(x_c_ref - x_c_raw), 0.05,
                f"{label} edge: refined x_c ({x_c_ref}) barely differs from "
                f"the raw peak ({x_c_raw}) even at high confidence -- the "
                f"wraparound-based correction does not appear to be exercised")

    @cpu_and_gpu
    def test_subpixel_peak_refine_matches_downstream_gain_correction_toggle_pattern(self, target_device_idx, xp):
        """
        Sanity check that subpixel_peak_refine composes correctly with the
        rest of the pipeline (not just x_c/y_c in isolation): with
        shrinkage neutralized (same recipe as
        test_subpixel_accuracy_when_shrinkage_neutralized), the emitted
        slope for a small injected sub-pixel shift must still track the
        true shift in sign and rough magnitude when refinement is enabled,
        i.e. enabling Step 1 refinement does not break Step 2/3's own
        correction chain.

        Confidence gate (2026-09-14): the gate reads w_smooth as it stood
        BEFORE this frame's own EMA update, which is exactly 0 for a fresh
        instance's very first frame -- without a warm-up, refinement would
        contribute nothing here regardless of whether it is enabled, making
        this test indistinguishable from subpixel_peak_refine=False. With
        w_ema_alpha=1.0 (no EMA lag) a single warm-up frame on this same
        high-SNR spot is enough to bring w_smooth to ~1 (confirmed directly)
        by the time the assertion frame runs.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5,
                                         k_wiener=1e-8, b_reg=0.0, ron_e=0.0,
                                         w_ema_alpha=1.0, subpixel_peak_refine=True,
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        shift_x, shift_y = 0.3, -0.2
        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                    shift_dx=shift_x, shift_dy=shift_y)
        self._run_frame(slopec, pixels, frame, t)
        w_smooth_before_assertion = float(cpuArray(slopec.w_smooth)[0])
        self.assertGreater(w_smooth_before_assertion, 0.999,
            "Test setup assumption violated: expected w_smooth to reach ~1 "
            "after a single high-SNR warm-up frame at w_ema_alpha=1.0")
        self._run_frame(slopec, pixels, frame, t * 2)

        slopes_x = cpuArray(slopec.outputs['out_slopes'].xslopes)
        slopes_y = cpuArray(slopec.outputs['out_slopes'].yslopes)
        expected_slope_x = shift_x / (subap_npx / 2.0)
        expected_slope_y = shift_y / (subap_npx / 2.0)

        np.testing.assert_allclose(slopes_x, expected_slope_x, atol=0.05,
                                   err_msg="X sub-pixel accuracy/sign failed with refinement enabled")
        np.testing.assert_allclose(slopes_y, expected_slope_y, atol=0.05,
                                   err_msg="Y sub-pixel accuracy/sign failed with refinement enabled")

    @cpu_and_gpu
    def test_subpixel_peak_refine_suppressed_at_zero_initial_confidence(self, target_device_idx, xp):
        """
        Confidence gate (2026-09-14, see class docstring's subpixel_peak_refine
        entry): the sub-pixel correction is scaled by self.w_smooth AS IT
        STANDS AT THE START OF THE FRAME. A freshly constructed instance has
        w_smooth = 0 exactly (see __init__), and the gate reads that value
        BEFORE this frame's own EMA update -- so on frame 1, regardless of
        w_ema_alpha, the correction must be suppressed to exactly zero even
        for a spot placed at a large, clearly-offset sub-pixel position that
        would otherwise (see
        test_subpixel_peak_refine_correction_scales_linearly_with_w_smooth)
        produce a large correction. Checked against a
        subpixel_peak_refine=False reference on the same frame: with the
        gate at exactly 0, the two must be bit-for-bit identical.

        A second scenario checks the OTHER route to near-zero confidence
        named in the docstring: w_ema_alpha set very small, run over several
        frames of the same bright spot so w_smooth barely moves off 0 even
        though it is no longer literally the frame-1 value.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        shift_x, shift_y = 0.3, -0.35
        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                    shift_dx=shift_x, shift_dy=shift_y)

        # --- Scenario 1: literal frame 1, default w_ema_alpha ---
        pixels_raw = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_ref = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec_raw = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                             subpixel_peak_refine=False,
                                             target_device_idx=target_device_idx)
        slopec_ref = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                             subpixel_peak_refine=True,
                                             target_device_idx=target_device_idx)
        slopec_raw.inputs['in_pixels'].set(pixels_raw)
        slopec_ref.inputs['in_pixels'].set(pixels_ref)

        self._run_frame(slopec_raw, pixels_raw, frame, t)
        self._run_frame(slopec_ref, pixels_ref, frame, t)

        x_c_raw = cpuArray(slopec_raw.outputs['out_x_c'].value)
        y_c_raw = cpuArray(slopec_raw.outputs['out_y_c'].value)
        x_c_ref = cpuArray(slopec_ref.outputs['out_x_c'].value)
        y_c_ref = cpuArray(slopec_ref.outputs['out_y_c'].value)

        np.testing.assert_array_equal(x_c_ref, x_c_raw,
            err_msg="Frame-1 correction was not exactly suppressed: out_x_c "
                    "differs from the unrefined value despite w_smooth "
                    "starting at 0")
        np.testing.assert_array_equal(y_c_ref, y_c_raw,
            err_msg="Frame-1 correction was not exactly suppressed: out_y_c "
                    "differs from the unrefined value despite w_smooth "
                    "starting at 0")

        # --- Scenario 2: tiny w_ema_alpha, several frames, w_smooth stays ~0 ---
        pixels_tiny = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec_tiny = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                              w_ema_alpha=1e-6, subpixel_peak_refine=True,
                                              target_device_idx=target_device_idx)
        slopec_tiny.inputs['in_pixels'].set(pixels_tiny)
        for i in range(1, 6):
            self._run_frame(slopec_tiny, pixels_tiny, frame, t * i)
        w_smooth_tiny = float(cpuArray(slopec_tiny.w_smooth)[0])
        self.assertLess(w_smooth_tiny, 1e-4,
            "Test setup assumption violated: w_ema_alpha=1e-6 should keep "
            "w_smooth extremely close to its zero initial value after only "
            "a few frames")

        x_c_tiny = float(cpuArray(slopec_tiny.outputs['out_x_c'].value)[0])
        y_c_tiny = float(cpuArray(slopec_tiny.outputs['out_y_c'].value)[0])
        self.assertAlmostEqual(x_c_tiny, float(x_c_raw[0]), places=3,
            msg="Correction was not suppressed with a near-zero w_smooth "
                "(tiny w_ema_alpha scenario)")
        self.assertAlmostEqual(y_c_tiny, float(y_c_raw[0]), places=3,
            msg="Correction was not suppressed with a near-zero w_smooth "
                "(tiny w_ema_alpha scenario)")

    @cpu_and_gpu
    def test_subpixel_peak_refine_correction_near_full_magnitude_when_warmed_up(self, target_device_idx, xp):
        """
        Complements the zero-confidence suppression check above: after a
        warm-up sequence that drives w_smooth (inspected directly, not
        assumed) close to 1, the applied correction (out_x_c/out_y_c minus
        the raw integer-pixel position) must be close to the FULL, un-gated
        magnitude -- i.e. what a directly-computed parabolic interpolation
        on the same correlation data would give, unscaled.

        The un-gated reference is obtained by forcing self.w_smooth to
        exactly 1.0 on a fresh instance right before a single frame: the
        gate reads w_smooth as it stands at the START of the frame (before
        that frame's own EMA update), so this yields exactly the
        full-strength correction with no need to re-derive the parabolic
        formula independently in the test.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        shift_x, shift_y = 0.3, -0.35
        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                    shift_dx=shift_x, shift_dy=shift_y)

        # Raw (unrefined) reference position.
        pixels_raw = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec_raw = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                             subpixel_peak_refine=False,
                                             target_device_idx=target_device_idx)
        slopec_raw.inputs['in_pixels'].set(pixels_raw)
        self._run_frame(slopec_raw, pixels_raw, frame, t)
        x_c_raw = float(cpuArray(slopec_raw.outputs['out_x_c'].value)[0])
        y_c_raw = float(cpuArray(slopec_raw.outputs['out_y_c'].value)[0])

        # Un-gated reference: force w_smooth to exactly 1.0 before a single frame.
        pixels_full = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec_full = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                              subpixel_peak_refine=True,
                                              target_device_idx=target_device_idx)
        slopec_full.inputs['in_pixels'].set(pixels_full)
        slopec_full.w_smooth[:] = 1.0
        self._run_frame(slopec_full, pixels_full, frame, t)
        correction_full_x = float(cpuArray(slopec_full.outputs['out_x_c'].value)[0]) - x_c_raw
        correction_full_y = float(cpuArray(slopec_full.outputs['out_y_c'].value)[0]) - y_c_raw
        self.assertGreater(abs(correction_full_x), 0.1,
            "Test setup assumption violated: expected a large un-gated x correction")

        # Warmed-up instance: default w_ema_alpha, many identical bright/shifted
        # frames so w_smooth ramps toward 1 via its own EMA (not forced).
        pixels_warm = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec_warm = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                              subpixel_peak_refine=True,
                                              target_device_idx=target_device_idx)
        slopec_warm.inputs['in_pixels'].set(pixels_warm)
        for i in range(1, 41):
            self._run_frame(slopec_warm, pixels_warm, frame, t * i)
        w_smooth_warm = float(cpuArray(slopec_warm.w_smooth)[0])
        self.assertGreater(w_smooth_warm, 0.999,
            "Test setup assumption violated: expected w_smooth to have "
            "ramped close to 1 after 40 identical high-SNR frames")

        correction_warm_x = float(cpuArray(slopec_warm.outputs['out_x_c'].value)[0]) - x_c_raw
        correction_warm_y = float(cpuArray(slopec_warm.outputs['out_y_c'].value)[0]) - y_c_raw

        self.assertAlmostEqual(correction_warm_x, correction_full_x, delta=0.005,
            msg="Warmed-up correction did not approach the full, un-gated magnitude (x)")
        self.assertAlmostEqual(correction_warm_y, correction_full_y, delta=0.005,
            msg="Warmed-up correction did not approach the full, un-gated magnitude (y)")

    @cpu_and_gpu
    def test_subpixel_peak_refine_correction_scales_linearly_with_w_smooth(self, target_device_idx, xp):
        """
        The actual new behaviour worth locking in: for a FIXED true
        sub-pixel offset, the magnitude of the correction actually applied
        must scale with whatever w_smooth value is in effect at that frame.
        self.w_smooth is forced directly to a range of levels on otherwise
        identical fresh instances/frames (the gate reads it before this
        frame's own EMA update, so forcing it beforehand deterministically
        sets the gate for that frame) -- since the underlying dx/dy from
        the 3-point parabolic fit is computed purely from the correlation
        map and does not itself depend on w_smooth, the applied correction
        must be EXACTLY proportional to the forced w_smooth level, not just
        "smaller when w_smooth is smaller".
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        shift_x, shift_y = 0.3, -0.35
        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e6, bg=0.0,
                                    shift_dx=shift_x, shift_dy=shift_y)

        pixels_raw = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec_raw = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                             subpixel_peak_refine=False,
                                             target_device_idx=target_device_idx)
        slopec_raw.inputs['in_pixels'].set(pixels_raw)
        self._run_frame(slopec_raw, pixels_raw, frame, t)
        x_c_raw = float(cpuArray(slopec_raw.outputs['out_x_c'].value)[0])

        forced_levels = [0.0, 0.2, 0.5, 0.8, 1.0]
        corrections_x = []
        for w_level in forced_levels:
            pixels_f = Pixels(*ccd_shape, target_device_idx=target_device_idx)
            slopec_f = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                               subpixel_peak_refine=True,
                                               target_device_idx=target_device_idx)
            slopec_f.inputs['in_pixels'].set(pixels_f)
            slopec_f.w_smooth[:] = w_level
            self._run_frame(slopec_f, pixels_f, frame, t)
            x_c_f = float(cpuArray(slopec_f.outputs['out_x_c'].value)[0])
            corrections_x.append(x_c_f - x_c_raw)

        # Exactly zero at w_smooth=0.
        self.assertAlmostEqual(corrections_x[0], 0.0, places=9,
            msg=f"Correction was not exactly 0 at w_smooth=0: {corrections_x[0]}")

        # Strictly increasing magnitude as the forced w_smooth level increases
        # (same sign throughout, since dx itself does not depend on w_smooth).
        for a, b in zip(corrections_x, corrections_x[1:]):
            self.assertLess(abs(a), abs(b),
                f"Correction magnitude did not increase monotonically with "
                f"w_smooth: {corrections_x} at levels {forced_levels}")

        # Exact linear proportionality: correction / w_smooth must be the
        # SAME constant at every nonzero level tested.
        ratios = [c / w for c, w in zip(corrections_x[1:], forced_levels[1:])]
        for r in ratios[1:]:
            self.assertAlmostEqual(r, ratios[0], places=6,
                msg=f"Correction did not scale exactly linearly with the "
                    f"forced w_smooth level: ratios={ratios}")

    @cpu_and_gpu
    def test_step1_fwhm_pix_default_matches_pre_existing_behaviour_bit_for_bit(self, target_device_idx, xp):
        """
        step1_fwhm_pix (2026-09-15, see class docstring) defaults to None,
        which falls back to fwhm_pix -- this must be bit-for-bit identical
        both to an instance built the OLD way (the parameter not passed at
        all, i.e. every pre-2026-09-15 call site) and to one where fwhm_pix
        is passed again explicitly as step1_fwhm_pix. Checks the three
        quantities the parameter can possibly influence (fft_template_conj,
        sigma_psf_sq, g_wcog) plus an end-to-end run of calc_slopes_nofor()
        on a varying synthetic frame sequence, not just constructor-time
        state, using the same "new knob is inert" pattern as
        test_subpixel_peak_refine_default_false_is_pure_no_op.
        """
        subap_npx, t = 16, int(1e9)
        subapdata_old, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_explicit, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_old = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_explicit = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        common = dict(fwhm_pix=1.5, k_wiener=10.0, b_reg=0.5, ron_e=1.0,
                     w_ema_alpha=0.2, target_device_idx=target_device_idx)
        slopec_old = AdaptiveShrinkageSlopec(subapdata_old, **common)  # no step1_fwhm_pix at all
        slopec_explicit = AdaptiveShrinkageSlopec(subapdata_explicit, step1_fwhm_pix=1.5, **common)
        slopec_old.inputs['in_pixels'].set(pixels_old)
        slopec_explicit.inputs['in_pixels'].set(pixels_explicit)

        self.assertEqual(slopec_old.step1_fwhm_pix, 1.5,
                         "step1_fwhm_pix did not fall back to fwhm_pix when left None")

        np.testing.assert_array_equal(cpuArray(slopec_explicit.fft_template_conj),
            cpuArray(slopec_old.fft_template_conj),
            err_msg="step1_fwhm_pix=fwhm_pix must reproduce the default-None "
                    "template bit-for-bit")
        self.assertAlmostEqual(slopec_explicit.sigma_psf_sq, slopec_old.sigma_psf_sq, places=9)
        self.assertAlmostEqual(slopec_explicit.g_wcog, slopec_old.g_wcog, places=9)

        rng = np.random.RandomState(2026)
        for i in range(1, 21):
            flux = float(rng.choice([0.0, 1.0, 10.0, 1e3, 1e5]))
            shift_dx = float(rng.uniform(-0.4, 0.4))
            shift_dy = float(rng.uniform(-0.4, 0.4))
            noise_std = float(rng.uniform(0.0, 1.5))
            frame = self.generate_spots(ccd_shape, subapdata_old, xp, flux=flux, bg=0.5,
                                        shift_dx=shift_dx, shift_dy=shift_dy,
                                        noise_std=noise_std)
            self._run_frame(slopec_old, pixels_old, frame, t * i)
            self._run_frame(slopec_explicit, pixels_explicit, frame, t * i)

            xo = cpuArray(slopec_old.outputs['out_slopes'].xslopes)
            xe = cpuArray(slopec_explicit.outputs['out_slopes'].xslopes)
            yo = cpuArray(slopec_old.outputs['out_slopes'].yslopes)
            ye = cpuArray(slopec_explicit.outputs['out_slopes'].yslopes)

            np.testing.assert_array_equal(xe, xo,
                err_msg=f"frame {i}: xslopes diverged between default (None) and "
                        f"explicit step1_fwhm_pix=fwhm_pix")
            np.testing.assert_array_equal(ye, yo,
                err_msg=f"frame {i}: yslopes diverged between default (None) and "
                        f"explicit step1_fwhm_pix=fwhm_pix")

    @cpu_and_gpu
    def test_step1_fwhm_pix_isolates_template_width_from_sigma_and_gwcog(self, target_device_idx, xp):
        """
        Core isolation property step1_fwhm_pix exists for (2026-09-15, see
        its docstring entry): overriding it to a value DIFFERENT from
        fwhm_pix must change the Step-1 template (fft_template_conj) while
        leaving sigma_psf_sq and g_wcog (with g_wcog=None) driven by
        fwhm_pix alone, unchanged. Checked directly against a third instance
        where fwhm_pix itself is set to step1_fwhm_pix's value: that
        instance's sigma_psf_sq/g_wcog DO differ, confirming the isolation
        is a genuine effect of this parameter, not something that would
        have come out equal anyway. wcog_fwhm_pix is pinned to a fixed value
        independent of fwhm_pix/step1_fwhm_pix: left at its own default
        (None) it also tracks fwhm_pix, which would make g_wcog identically
        0.5 regardless of fwhm_pix (sig_w == sig_s always) and mask the
        very difference this test needs to see in slopec_c.
        """
        subap_npx = 16
        subapdata_a, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_b, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_c, _ = self.get_test_setup(target_device_idx, xp, subap_npx)

        slopec_a = AdaptiveShrinkageSlopec(subapdata_a, fwhm_pix=1.5, wcog_fwhm_pix=2.0,
                                           target_device_idx=target_device_idx)
        slopec_b = AdaptiveShrinkageSlopec(subapdata_b, fwhm_pix=1.5, step1_fwhm_pix=3.0,
                                           wcog_fwhm_pix=2.0, target_device_idx=target_device_idx)
        slopec_c = AdaptiveShrinkageSlopec(subapdata_c, fwhm_pix=3.0, wcog_fwhm_pix=2.0,
                                           target_device_idx=target_device_idx)

        # Template: B (step1 override) must match C (fwhm_pix set directly to
        # the same width) bit-for-bit -- both build the Step-1 core at width
        # 3.0 -- and must differ from A (pure fwhm_pix=1.5, step1 falls back to it).
        np.testing.assert_array_equal(cpuArray(slopec_b.fft_template_conj),
            cpuArray(slopec_c.fft_template_conj),
            err_msg="step1_fwhm_pix=3.0 did not build the same Step-1 template "
                    "as an instance with fwhm_pix=3.0 directly")
        self.assertFalse(np.array_equal(cpuArray(slopec_b.fft_template_conj),
                                        cpuArray(slopec_a.fft_template_conj)),
            "step1_fwhm_pix=3.0 did not change the Step-1 template relative "
            "to fwhm_pix=1.5 alone")

        # sigma_psf_sq/g_wcog: B must match A (both effectively fwhm_pix=1.5
        # for this purpose), NOT C (which genuinely has fwhm_pix=3.0) -- the
        # isolation property under test.
        self.assertAlmostEqual(slopec_b.sigma_psf_sq, slopec_a.sigma_psf_sq, places=9,
                               msg="sigma_psf_sq changed with step1_fwhm_pix -- it "
                                   "must stay tied to fwhm_pix alone")
        self.assertAlmostEqual(slopec_b.g_wcog, slopec_a.g_wcog, places=9,
                               msg="g_wcog changed with step1_fwhm_pix -- it must "
                                   "stay tied to fwhm_pix alone (when g_wcog=None)")
        self.assertNotAlmostEqual(slopec_c.sigma_psf_sq, slopec_a.sigma_psf_sq, places=6,
            msg="Test setup assumption violated: sigma_psf_sq should genuinely "
                "differ when fwhm_pix itself changes")
        self.assertNotAlmostEqual(slopec_c.g_wcog, slopec_a.g_wcog, places=6,
            msg="Test setup assumption violated: g_wcog should genuinely "
                "differ when fwhm_pix itself changes")

    @cpu_and_gpu
    def test_step1_fwhm_pix_narrower_template_sharpens_correlation_margin(self, target_device_idx, xp):
        """
        The physical property step1_fwhm_pix was added for (see its
        docstring: "a template narrower than the true PSF sharpens the
        margin" between the winning grid hypothesis and its runner-up): for
        a fixed, noiseless synthetic spot, the noiseless Step-1 correlation
        margin (argmax bin height minus the next-highest bin, as a fraction
        of the peak) must strictly increase as step1_fwhm_pix narrows, all
        else (fwhm_pix, the true spot) held fixed. Checked directly on the
        raw FFT correlation (fft_pix * fft_template_conj), bypassing the
        spatial-prior weighting Step 1 applies before its own argmax, since
        the prior is identical across the step1_fwhm_pix values compared and
        would only dilute the effect under test.
        """
        subap_npx = 16
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)

        def margin_for(step1_fwhm_pix):
            slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5,
                                             step1_fwhm_pix=step1_fwhm_pix,
                                             target_device_idx=target_device_idx)
            frame = self.generate_spots(ccd_shape, subapdata, xp, fwhm=1.5, flux=100.0,
                                        shift_dx=0.0, shift_dy=0.0)
            fft_pix = xp.fft.fft2(frame[None, :, :], axes=(1, 2))
            corr = cpuArray(xp.fft.ifft2(fft_pix * slopec.fft_template_conj,
                                         axes=(1, 2)).real[0])
            flat = corr.reshape(-1)
            order = np.argsort(flat)[::-1]
            peak, runner_up = flat[order[0]], flat[order[1]]
            return (peak - runner_up) / peak

        step1_values = [4.0, 2.0, 1.0]  # widest -> narrowest
        margins = [margin_for(v) for v in step1_values]

        for wider_margin, narrower_margin in zip(margins[:-1], margins[1:]):
            self.assertGreater(narrower_margin, wider_margin,
                f"Correlation margin did not increase for a narrower "
                f"step1_fwhm_pix: {list(zip(step1_values, margins))}")

    @cpu_and_gpu
    def test_step1_fwhm_pix_with_halo_only_widens_the_core_component(self, target_device_idx, xp):
        """
        Interaction with the 2026-09-13 two-component template (see
        halo_fwhm_pix/halo_fraction docstrings): step1_fwhm_pix must apply
        to the CORE component's width only, leaving the halo component at
        its own independent halo_fwhm_pix. Checked the same way as
        test_step1_fwhm_pix_isolates_template_width_from_sigma_and_gwcog --
        an instance with step1_fwhm_pix=3.0 (core width) + halo_fwhm_pix=6.0
        must produce a template bit-for-bit identical to one with
        fwhm_pix=3.0 directly (same core width) + the same halo_fwhm_pix/
        halo_fraction, while still keeping its OWN sigma_psf_sq/g_wcog tied
        to its own (unrelated) fwhm_pix=1.5, not to step1_fwhm_pix or to the
        halo width.
        """
        subap_npx = 16
        subapdata_a, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_b, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_c, _ = self.get_test_setup(target_device_idx, xp, subap_npx)

        halo_kwargs = dict(halo_fwhm_pix=6.0, halo_fraction=0.3,
                           target_device_idx=target_device_idx)
        slopec_step1 = AdaptiveShrinkageSlopec(subapdata_a, fwhm_pix=1.5,
                                               step1_fwhm_pix=3.0, **halo_kwargs)
        slopec_core_direct = AdaptiveShrinkageSlopec(subapdata_b, fwhm_pix=3.0,
                                                     **halo_kwargs)

        np.testing.assert_array_equal(cpuArray(slopec_step1.fft_template_conj),
            cpuArray(slopec_core_direct.fft_template_conj),
            err_msg="step1_fwhm_pix did not apply to the core component's width "
                    "alone: the two-component template differs from an instance "
                    "with fwhm_pix set directly to the same core width")

        # Sanity check that the two-component mix was not silently broken by
        # the override: still integrates to 1, exactly like the plain
        # halo_fwhm_pix/halo_fraction case (see
        # test_two_component_template_stays_normalized).
        conj_fft = xp.conj(slopec_step1.fft_template_conj)
        template = cpuArray(xp.fft.ifft2(conj_fft, axes=(1, 2)).real[0])
        self.assertAlmostEqual(float(np.sum(template)), 1.0, places=5,
            msg="Two-component template with step1_fwhm_pix override does not "
                "integrate to 1")

        # sigma_psf_sq/g_wcog stay tied to this instance's OWN fwhm_pix=1.5,
        # not to step1_fwhm_pix=3.0 or to halo_fwhm_pix=6.0.
        slopec_reference = AdaptiveShrinkageSlopec(subapdata_c, fwhm_pix=1.5,
                                                    **halo_kwargs)
        self.assertAlmostEqual(slopec_step1.sigma_psf_sq, slopec_reference.sigma_psf_sq, places=9,
            msg="sigma_psf_sq changed with step1_fwhm_pix under the "
                "two-component template")
        self.assertAlmostEqual(slopec_step1.g_wcog, slopec_reference.g_wcog, places=9,
            msg="g_wcog changed with step1_fwhm_pix under the "
                "two-component template")

    # =====================================================================
    # gate_type/gate_params (2026-09-18): pluggable Step-3 confidence-gate
    # strategy abstraction, replacing the earlier relative_gate_enable/
    # relative_gate_* flag family. See specula/lib/confidence_gates.py and
    # the class docstring's gate_type entry for the full design rationale.
    # =====================================================================

    @cpu_and_gpu
    def test_relative_gate_disabled_is_bit_for_bit_inert_and_matches_classic_formula(self, target_device_idx, xp):
        """
        gate_type='wiener' (the default) must be exactly the pre-2026-09-17
        behaviour: out_margin stays identically 0 (WienerGate.needs_margin is
        False) and out_rho_sq_ceiling stays identically 0 (WienerGate.telemetry()
        returns {}, so that output is never written) across several varying
        frames, REGARDLESS of what gate_params is set to alongside
        gate_type='wiener' -- build_confidence_gate() ignores gate_params
        entirely for the 'wiener' class (same "new knob is inert" pattern as
        test_stuck_detector_and_both_fixes_default_inert /
        test_subpixel_peak_refine_default_false_is_pure_no_op) -- an
        aggressive-but-wiener instance must emit bit-for-bit identical
        slopes to the plain default instance.

        Separately, with the EMA lag removed (w_ema_alpha=1.0, so
        w_out == w_raw exactly after one frame), the emitted gain must equal
        the classic absolute formula rho_sq / (rho_sq + k_wiener) computed
        independently here from the class's own out_rho_sq telemetry --
        pinning down that the wiener-gate path truly depends only on
        k_wiener, not on anything relative-gate-specific.
        """
        subap_npx, t = 16, int(1e9)
        subapdata_default, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_aggressive, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_default = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_aggressive = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        common = dict(fwhm_pix=1.5, k_wiener=10.0, ron_e=1.0, w_ema_alpha=0.2,
                     target_device_idx=target_device_idx)
        slopec_default = AdaptiveShrinkageSlopec(subapdata_default, **common)
        # gate_type explicitly 'wiener' with gate_params that WOULD visibly
        # fire if they were forwarded to a margin/ceiling-based gate --
        # build_confidence_gate() ignores gate_params entirely for 'wiener'.
        slopec_aggressive = AdaptiveShrinkageSlopec(
            subapdata_aggressive,
            gate_type='wiener',
            gate_params={'k_rel': 100.0, 'margin_thresh': -1.0,
                         'ema_alpha': 1.0, 'ceiling_init': 999.0},
            margin_exclude_radius_px=0.0,
            **common)
        slopec_default.inputs['in_pixels'].set(pixels_default)
        slopec_aggressive.inputs['in_pixels'].set(pixels_aggressive)

        rng = np.random.RandomState(2026_09_17)
        for i in range(1, 11):
            flux = float(rng.choice([0.0, 1.0, 10.0, 1e3, 1e5]))
            shift_dx = float(rng.uniform(-0.4, 0.4))
            shift_dy = float(rng.uniform(-0.4, 0.4))
            noise_std = float(rng.uniform(0.0, 1.5))
            frame = self.generate_spots(ccd_shape, subapdata_default, xp, flux=flux, bg=0.5,
                                        shift_dx=shift_dx, shift_dy=shift_dy,
                                        noise_std=noise_std)
            self._run_frame(slopec_default, pixels_default, frame, t * i)
            self._run_frame(slopec_aggressive, pixels_aggressive, frame, t * i)

            for slopec, label in ((slopec_default, "default"), (slopec_aggressive, "aggressive-but-wiener")):
                # out_margin is only ever written under "if
                # self._gate.needs_margin", False for WienerGate -- checking
                # it is identically 0 doubles as confirming that path never runs.
                margin = cpuArray(slopec.outputs['out_margin'].value)
                np.testing.assert_array_equal(margin, np.zeros_like(margin),
                    err_msg=f"frame {i} ({label}): out_margin is not identically 0 "
                            f"with gate_type='wiener'")
                # out_rho_sq_ceiling is only written when the gate's own
                # telemetry() dict contains 'rho_sq_ceiling' -- WienerGate's
                # telemetry() is always {}, so this must stay at its
                # zero-initialized default regardless of gate_params.
                ceiling = cpuArray(slopec.outputs['out_rho_sq_ceiling'].value)
                np.testing.assert_array_equal(ceiling, np.zeros_like(ceiling),
                    err_msg=f"frame {i} ({label}): out_rho_sq_ceiling is not identically 0 "
                            f"with gate_type='wiener' (WienerGate.telemetry() should be empty)")

            xd = cpuArray(slopec_default.outputs['out_slopes'].xslopes)
            xa = cpuArray(slopec_aggressive.outputs['out_slopes'].xslopes)
            yd = cpuArray(slopec_default.outputs['out_slopes'].yslopes)
            ya = cpuArray(slopec_aggressive.outputs['out_slopes'].yslopes)
            np.testing.assert_allclose(xa, xd, atol=1e-9, rtol=0,
                err_msg=f"frame {i}: xslopes diverged between default and "
                        f"aggressive-but-wiener instances -- gate_params is not "
                        f"fully ignored for gate_type='wiener'")
            np.testing.assert_allclose(ya, yd, atol=1e-9, rtol=0,
                err_msg=f"frame {i}: yslopes diverged between default and "
                        f"aggressive-but-wiener instances -- gate_params is not "
                        f"fully ignored for gate_type='wiener'")

        # Classic absolute-formula check, EMA lag removed.
        subapdata_formula, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_formula = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec_formula = AdaptiveShrinkageSlopec(subapdata_formula, fwhm_pix=1.5, k_wiener=10.0,
                                                 ron_e=1.0, w_ema_alpha=1.0,
                                                 target_device_idx=target_device_idx)
        slopec_formula.inputs['in_pixels'].set(pixels_formula)
        frame = self.generate_spots(ccd_shape, subapdata_formula, xp, flux=200.0, bg=0.5, noise_std=0.5)
        self._run_frame(slopec_formula, pixels_formula, frame, t)

        rho_sq = cpuArray(slopec_formula.outputs['out_rho_sq'].value)
        expected_w = rho_sq / (rho_sq + 10.0)
        actual_w = cpuArray(slopec_formula.w_out)
        np.testing.assert_allclose(actual_w, expected_w, atol=1e-6,
            err_msg="Default gate_type='wiener' did not reproduce the classic "
                    "absolute formula rho_sq / (rho_sq + k_wiener)")

    @cpu_and_gpu
    def test_relative_gate_ceiling_init_defaults_to_k_wiener_or_explicit_value(self, target_device_idx, xp):
        """
        self._gate.rho_sq_ceiling (see RelativeCeilingGate's ceiling_init
        docstring) must equal k_wiener itself at construction when
        ceiling_init is left at its default None, for any k_wiener value --
        and must equal the explicit value when one is passed via
        gate_params, regardless of k_wiener. Checked before any frame is
        run (construction-time state only).
        """
        subap_npx = 16
        subapdata_a, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_b, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_c, _ = self.get_test_setup(target_device_idx, xp, subap_npx)

        slopec_default_k10 = AdaptiveShrinkageSlopec(subapdata_a, gate_type='relative_ceiling',
                                                     k_wiener=10.0, target_device_idx=target_device_idx)
        slopec_default_k25 = AdaptiveShrinkageSlopec(subapdata_b, gate_type='relative_ceiling',
                                                     k_wiener=25.0, target_device_idx=target_device_idx)
        slopec_explicit = AdaptiveShrinkageSlopec(subapdata_c, gate_type='relative_ceiling',
                                                  k_wiener=10.0, gate_params={'ceiling_init': 7.5},
                                                  target_device_idx=target_device_idx)

        np.testing.assert_array_equal(cpuArray(slopec_default_k10._gate.rho_sq_ceiling),
            np.full(subapdata_a.n_subaps, 10.0),
            err_msg="rho_sq_ceiling did not default-initialise to k_wiener=10.0")
        np.testing.assert_array_equal(cpuArray(slopec_default_k25._gate.rho_sq_ceiling),
            np.full(subapdata_b.n_subaps, 25.0),
            err_msg="rho_sq_ceiling did not default-initialise to k_wiener=25.0")
        np.testing.assert_array_equal(cpuArray(slopec_explicit._gate.rho_sq_ceiling),
            np.full(subapdata_c.n_subaps, 7.5),
            err_msg="rho_sq_ceiling did not honour an explicit "
                    "ceiling_init, using k_wiener instead")

    @cpu_and_gpu
    def test_relative_gate_ceiling_updates_only_on_high_margin_frames(self, target_device_idx, xp):
        """
        rho_sq_ceiling must update by the documented EMA
        ((1-alpha)*ceiling + alpha*rho_sq) exactly on a frame whose OWN
        margin exceeds the gate's own margin_thresh, using that SAME frame's
        own rho_sq (Step 1's margin and Step 3's rho_sq/ceiling-update both
        run on the same frame's pixel data -- there is no one-frame lag),
        and must stay EXACTLY unchanged on a frame whose margin does not
        clear the threshold. Checked algebraically frame-by-frame using the
        class's own out_margin/out_rho_sq telemetry, not assumed.

        A single bright, clean, centred spot gives a high margin (~0.9+,
        matches the manual sanity check in the handoff); two equal-height,
        well-separated Gaussian spots give a genuine tie (margin ~0) -- the
        same construction as the dedicated tied-peak edge case below.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        np_sub = subapdata.np_sub

        def two_spot_frame(spots):
            """spots: list of (flux, shift_dx, shift_dy) summed into one
            np_sub x np_sub frame (single sub-aperture covering the whole
            array, as in the default get_test_setup())."""
            cntrd = (np_sub - 1) / 2.0
            xg = np.arange(np_sub) - cntrd
            yg = np.arange(np_sub) - cntrd
            xx0, yy0 = np.meshgrid(xg, yg)
            sigma = 1.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            ccd = np.zeros((np_sub, np_sub), dtype=np.float32)
            for flux, shift_dx, shift_dy in spots:
                xx = xx0 - shift_dx
                yy = yy0 - shift_dy
                gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                gaussian = (gaussian / np.sum(gaussian)) * flux
                ccd += gaussian
            return xp.asarray(ccd)

        bright_frame = two_spot_frame([(1e4, 0.0, 0.0)])
        tie_frame = two_spot_frame([(500.0, -4.0, 0.0), (500.0, 4.0, 0.0)])

        margin_thresh = 0.5
        ema_alpha = 0.05
        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, k_wiener=10.0, ron_e=1.0,
                                         gate_type='relative_ceiling',
                                         gate_params={'margin_thresh': margin_thresh,
                                                      'ema_alpha': ema_alpha},
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        ceiling_before = float(cpuArray(slopec._gate.rho_sq_ceiling)[0])
        self.assertAlmostEqual(ceiling_before, 10.0, places=9)

        # Frame 1: bright, clean, high-margin.
        self._run_frame(slopec, pixels, bright_frame, t * 1)
        margin1 = float(cpuArray(slopec.outputs['out_margin'].value)[0])
        rho_sq1 = float(cpuArray(slopec.outputs['out_rho_sq'].value)[0])
        ceiling1 = float(cpuArray(slopec._gate.rho_sq_ceiling)[0])
        self.assertGreater(margin1, margin_thresh,
                           "Test setup assumption violated: expected a high-margin frame")
        expected_ceiling1 = (1.0 - ema_alpha) * ceiling_before + ema_alpha * rho_sq1
        self.assertAlmostEqual(ceiling1, expected_ceiling1, places=5,
            msg="rho_sq_ceiling did not update by the documented EMA on a "
                "high-margin frame")
        self.assertNotAlmostEqual(ceiling1, ceiling_before, places=6,
            msg="Test setup assumption violated: ceiling should have moved "
                "measurably away from its init value")

        # Frame 2: genuine tie, low margin -- ceiling must stay exactly put.
        self._run_frame(slopec, pixels, tie_frame, t * 2)
        margin2 = float(cpuArray(slopec.outputs['out_margin'].value)[0])
        ceiling2 = float(cpuArray(slopec._gate.rho_sq_ceiling)[0])
        self.assertLess(margin2, margin_thresh,
                        "Test setup assumption violated: expected a low-margin (tied) frame")
        self.assertEqual(ceiling2, ceiling1,
                         "rho_sq_ceiling changed on a below-threshold-margin frame")

        # Frame 3: bright again -- ceiling must update again, this time
        # relative to ceiling2 (== ceiling1), using frame 3's own rho_sq.
        self._run_frame(slopec, pixels, bright_frame, t * 3)
        margin3 = float(cpuArray(slopec.outputs['out_margin'].value)[0])
        rho_sq3 = float(cpuArray(slopec.outputs['out_rho_sq'].value)[0])
        ceiling3 = float(cpuArray(slopec._gate.rho_sq_ceiling)[0])
        self.assertGreater(margin3, margin_thresh)
        expected_ceiling3 = (1.0 - ema_alpha) * ceiling2 + ema_alpha * rho_sq3
        self.assertAlmostEqual(ceiling3, expected_ceiling3, places=5,
            msg="rho_sq_ceiling did not update by the documented EMA on a "
                "second high-margin frame")

    @cpu_and_gpu
    def test_relative_gate_ceiling_updates_independently_per_subaperture(self, target_device_idx, xp):
        """
        Reproduces the manual multi-subaperture finding in the handoff: in a
        2x2 sub-aperture grid, each sub-aperture's rho_sq_ceiling must update
        (or not) based purely on its OWN margin, independent of the other
        three. Deterministic construction (no noise, no reliance on flux-
        driven SNR variability, hence no flakiness): sub-apertures 0 and 1
        each get a single bright, clean, centred spot (high margin); 2 gets
        two equal-height, well-separated spots (a genuine tie, margin ~0);
        3 gets no spot at all (zero flux, margin exactly 0 -- the
        best_val==0 branch). Only 0 and 1's ceilings should move.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx, n_sub_side=2)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        self.assertEqual(subapdata.n_subaps, 4)
        np_sub = subapdata.np_sub

        def build_multi_subap_frame(spot_specs):
            """spot_specs: {subap_index: [(flux, shift_dx, shift_dy), ...]}.
            Missing indices get an all-zero region. Mirrors generate_spots()'s
            per-subaperture embedding via subapdata.idxs, but allows a
            different spot configuration per sub-aperture."""
            ccd = np.zeros(ccd_shape, dtype=np.float32)
            cntrd = (np_sub - 1) / 2.0
            xg = np.arange(np_sub) - cntrd
            yg = np.arange(np_sub) - cntrd
            xx0, yy0 = np.meshgrid(xg, yg)
            sigma = 1.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            for k in range(subapdata.n_subaps):
                idx_1d = cpuArray(subapdata.idxs[k])
                iy, ix = np.unravel_index(idx_1d, ccd_shape)
                min_y, max_y = np.min(iy), np.max(iy) + 1
                min_x, max_x = np.min(ix), np.max(ix) + 1
                for flux, shift_dx, shift_dy in spot_specs.get(k, []):
                    xx = xx0 - shift_dx
                    yy = yy0 - shift_dy
                    gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                    gaussian = (gaussian / np.sum(gaussian)) * flux
                    ccd[min_y:max_y, min_x:max_x] += gaussian
            return xp.asarray(ccd)

        spot_specs = {
            0: [(1000.0, 0.0, 0.0)],
            1: [(1000.0, 0.0, 0.0)],
            2: [(500.0, -3.0, 0.0), (500.0, 3.0, 0.0)],
            3: [],  # zero flux
        }
        frame = build_multi_subap_frame(spot_specs)

        k_wiener = 10.0
        margin_thresh = 0.5
        ema_alpha = 0.05
        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, k_wiener=k_wiener, ron_e=0.0,
                                         gate_type='relative_ceiling',
                                         gate_params={'margin_thresh': margin_thresh,
                                                      'ema_alpha': ema_alpha},
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        self._run_frame(slopec, pixels, frame, t)

        margin = cpuArray(slopec.outputs['out_margin'].value)
        rho_sq = cpuArray(slopec.outputs['out_rho_sq'].value)
        ceiling = cpuArray(slopec._gate.rho_sq_ceiling)

        self.assertGreater(margin[0], margin_thresh, "subap 0 (bright) should have a high margin")
        self.assertGreater(margin[1], margin_thresh, "subap 1 (bright) should have a high margin")
        self.assertLessEqual(margin[2], margin_thresh, "subap 2 (tie) should have a low margin")
        self.assertEqual(margin[3], 0.0, "subap 3 (zero flux) should have margin exactly 0")

        expected_ceiling_0 = (1.0 - ema_alpha) * k_wiener + ema_alpha * rho_sq[0]
        expected_ceiling_1 = (1.0 - ema_alpha) * k_wiener + ema_alpha * rho_sq[1]
        self.assertAlmostEqual(ceiling[0], expected_ceiling_0, places=5,
            msg="subap 0's ceiling did not update per its own high margin")
        self.assertAlmostEqual(ceiling[1], expected_ceiling_1, places=5,
            msg="subap 1's ceiling did not update per its own high margin")
        self.assertEqual(ceiling[2], k_wiener,
                         "subap 2's ceiling moved despite a low (tied) margin -- "
                         "not independent of the bright sub-apertures")
        self.assertEqual(ceiling[3], k_wiener,
                         "subap 3's ceiling moved despite zero flux/margin -- "
                         "not independent of the bright sub-apertures")

    @cpu_and_gpu
    def test_relative_gate_gain_formula_differs_from_absolute_gate(self, target_device_idx, xp):
        """
        For the SAME frame (hence the same rho_sq, verified directly), the
        relative gate's emitted gain must equal
        rho_sq / (rho_sq + k_rel * rho_sq_ceiling), computed
        independently here, and the absolute gate's must equal
        rho_sq / (rho_sq + k_wiener) -- and, with the ceiling driven well
        above k_wiener, the relative gate's gain must come out strictly
        LOWER (a bigger ceiling is a stricter relative bar), matching the
        sign derived from the two formulas rather than assumed.

        Uses w_ema_alpha=1.0 (no EMA lag, so w_out == w_raw exactly) and
        gate_params={'ema_alpha': 1.0} during a warm-up frame so the ceiling is
        set to EXACTLY that warm-up frame's own rho_sq (a known, very high
        value). The comparison frame is a genuine tie (margin ~0, verified
        below threshold), so the ceiling is guaranteed frozen at the warm-up
        value while w_raw is computed for the comparison frame.
        """
        subap_npx, t = 16, int(1e9)
        subapdata_abs, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_rel, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_abs = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_rel = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        np_sub = subapdata_abs.np_sub

        def two_spot_frame(spots):
            cntrd = (np_sub - 1) / 2.0
            xg = np.arange(np_sub) - cntrd
            yg = np.arange(np_sub) - cntrd
            xx0, yy0 = np.meshgrid(xg, yg)
            sigma = 1.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            ccd = np.zeros((np_sub, np_sub), dtype=np.float32)
            for flux, shift_dx, shift_dy in spots:
                xx = xx0 - shift_dx
                yy = yy0 - shift_dy
                gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                gaussian = (gaussian / np.sum(gaussian)) * flux
                ccd += gaussian
            return xp.asarray(ccd)

        k_wiener = 10.0
        k_rel = 0.3
        common = dict(fwhm_pix=1.5, k_wiener=k_wiener, ron_e=1.0, w_ema_alpha=1.0,
                     target_device_idx=target_device_idx)
        slopec_abs = AdaptiveShrinkageSlopec(subapdata_abs, **common)
        slopec_rel = AdaptiveShrinkageSlopec(subapdata_rel, gate_type='relative_ceiling',
                                             gate_params={'k_rel': k_rel,
                                                          'margin_thresh': 0.5,
                                                          'ema_alpha': 1.0},
                                             **common)
        slopec_abs.inputs['in_pixels'].set(pixels_abs)
        slopec_rel.inputs['in_pixels'].set(pixels_rel)

        # Warm-up: single bright, clean, high-margin spot -- with
        # ema_alpha=1.0 this sets rho_sq_ceiling to EXACTLY this frame's own
        # rho_sq.
        warmup_frame = two_spot_frame([(1e6, 0.0, 0.0)])
        self._run_frame(slopec_rel, pixels_rel, warmup_frame, t * 1)
        warmup_margin = float(cpuArray(slopec_rel.outputs['out_margin'].value)[0])
        self.assertGreater(warmup_margin, 0.5,
                           "Test setup assumption violated: expected a high-margin warm-up frame")
        ceiling_frozen = float(cpuArray(slopec_rel._gate.rho_sq_ceiling)[0])
        self.assertGreater(ceiling_frozen, 100.0 * k_wiener,
                           "Test setup assumption violated: expected the warm-up ceiling "
                           "to be driven well above k_wiener")

        # Comparison frame: a genuine tie, fed identically to both instances.
        tie_frame = two_spot_frame([(500.0, -4.0, 0.0), (500.0, 4.0, 0.0)])
        self._run_frame(slopec_abs, pixels_abs, tie_frame, t * 2)
        self._run_frame(slopec_rel, pixels_rel, tie_frame, t * 2)

        comparison_margin = float(cpuArray(slopec_rel.outputs['out_margin'].value)[0])
        self.assertLess(comparison_margin, 0.5,
                        "Test setup assumption violated: expected the comparison "
                        "frame's margin to stay below threshold (ceiling must not "
                        "move on this frame)")
        ceiling_after_comparison = float(cpuArray(slopec_rel._gate.rho_sq_ceiling)[0])
        self.assertEqual(ceiling_after_comparison, ceiling_frozen,
                         "rho_sq_ceiling moved on the (low-margin) comparison frame -- "
                         "it is no longer frozen at the warm-up value")

        rho_sq_abs = float(cpuArray(slopec_abs.outputs['out_rho_sq'].value)[0])
        rho_sq_rel = float(cpuArray(slopec_rel.outputs['out_rho_sq'].value)[0])
        np.testing.assert_allclose(rho_sq_rel, rho_sq_abs, rtol=1e-6,
            err_msg="rho_sq differs between the absolute- and relative-gate "
                    "instances on the identical comparison frame -- rho_sq "
                    "itself must not depend on gate_type")

        expected_w_abs = rho_sq_abs / (rho_sq_abs + k_wiener)
        expected_w_rel = rho_sq_rel / (rho_sq_rel + k_rel * ceiling_frozen)
        actual_w_abs = float(cpuArray(slopec_abs.w_out)[0])
        actual_w_rel = float(cpuArray(slopec_rel.w_out)[0])

        self.assertAlmostEqual(actual_w_abs, expected_w_abs, places=6,
            msg="Absolute-gate w_out did not match rho_sq / (rho_sq + k_wiener)")
        self.assertAlmostEqual(actual_w_rel, expected_w_rel, places=6,
            msg="Relative-gate w_out did not match rho_sq / (rho_sq + "
                "k_rel * rho_sq_ceiling)")
        self.assertLess(actual_w_rel, actual_w_abs,
            "Relative gate did not give a strictly lower gain than the "
            "absolute gate despite a ceiling driven far above k_wiener")

    @cpu_and_gpu
    def test_relative_gate_all_dark_frame_gives_zero_margin_no_nan(self, target_device_idx, xp):
        """
        Edge case explicitly called out in the handoff: a literal all-dark
        (zero flux, zero noise, zero background) frame with
        gate_type='relative_ceiling' must not crash or emit NaN/Inf, and must
        give margin exactly 0 (the best_val==0 branch, guarding the division
        that would otherwise be 0/0) -- this exercises the div-by-zero fix
        already applied to the new code (an xp.where-based safe denominator).
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         gate_type='relative_ceiling',
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        zero_frame = xp.zeros(ccd_shape, dtype=xp.float32)
        try:
            self._run_frame(slopec, pixels, zero_frame, t)
        except Exception as e:  # pragma: no cover - failure path
            self.fail(f"AdaptiveShrinkageSlopec raised on an all-dark frame "
                      f"with gate_type='relative_ceiling': {e!r}")

        margin = cpuArray(slopec.outputs['out_margin'].value)
        xslopes = cpuArray(slopec.outputs['out_slopes'].xslopes)
        yslopes = cpuArray(slopec.outputs['out_slopes'].yslopes)
        w_out = cpuArray(slopec.w_out)
        ceiling = cpuArray(slopec.outputs['out_rho_sq_ceiling'].value)

        np.testing.assert_array_equal(margin, np.zeros_like(margin),
            err_msg="margin was not exactly 0 on an all-dark frame")
        self.assertTrue(np.all(np.isfinite(xslopes)) and np.all(np.isfinite(yslopes)),
                        "NaN/Inf in emitted slopes on an all-dark frame")
        self.assertTrue(np.all(np.isfinite(w_out)), "NaN/Inf in w_out on an all-dark frame")
        self.assertTrue(np.all(np.isfinite(ceiling)), "NaN/Inf in rho_sq_ceiling on an all-dark frame")

    @cpu_and_gpu
    def test_relative_gate_tied_peaks_give_near_zero_margin(self, target_device_idx, xp):
        """
        Two exactly-equal-height, well-separated Gaussian spots are a
        genuine tie: Step 1's own winning peak and the best competing value
        elsewhere (outside the exclusion disk) are then (numerically) the
        same value, so margin = (best - second) / |best| must land at or
        very near 0 -- reproducing the manual sanity check in the handoff.
        """
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        np_sub = subapdata.np_sub

        cntrd = (np_sub - 1) / 2.0
        xg = np.arange(np_sub) - cntrd
        yg = np.arange(np_sub) - cntrd
        xx0, yy0 = np.meshgrid(xg, yg)
        sigma = 1.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        ccd = np.zeros((np_sub, np_sub), dtype=np.float32)
        for shift_dx in (-4.0, 4.0):
            xx = xx0 - shift_dx
            gaussian = np.exp(-(xx**2 + yy0**2) / (2 * sigma**2))
            gaussian = (gaussian / np.sum(gaussian)) * 500.0
            ccd += gaussian
        tie_frame = xp.asarray(ccd)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                         gate_type='relative_ceiling',
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        self._run_frame(slopec, pixels, tie_frame, t)

        margin = cpuArray(slopec.outputs['out_margin'].value)
        self.assertTrue(np.all(np.abs(margin) < 0.05),
                        f"Expected a near-zero margin for a genuine tie, got {margin}")

    @cpu_and_gpu
    def test_relative_gate_margin_exclude_radius_treats_near_bump_as_same_lobe(self, target_device_idx, xp):
        """
        margin_exclude_radius_px behaviour (see its docstring):
        a competing bump placed WITHIN the exclusion radius of the winning
        peak must be treated as part of that peak's own shoulder (excluded
        from the margin search), while one placed OUTSIDE it must be counted
        as a genuine competitor.

        Baseline: a single spot alone gives some margin. Adding a lower
        (but non-trivial) competing bump 1px away (inside the default 3.0px
        exclusion radius) must leave the margin close to baseline; the same
        bump placed 4.5px away (outside) must pull the margin down clearly.

        The "inside" offset is chosen close to the winning peak (not right
        at the exclusion boundary): a competing Gaussian bump's OWN
        correlation footprint has a width comparable to the exclusion
        radius itself, so a bump sitting right at ~2-3px still leaks a
        measurable tail past the boundary and is partially counted --
        this is an inherent property of a fixed-radius exclusion disk
        applied to a smooth (not delta-function) correlation feature, not a
        bug in the exclusion logic, but it means the clean "fully excluded"
        regime is only reached well inside the radius (verified empirically
        at 1px: margin within ~0.02 of baseline).
        """
        subap_npx, t = 16, int(1e9)
        np_sub = subap_npx

        def margin_for(offsets_and_fluxes):
            subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
            pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
            cntrd = (np_sub - 1) / 2.0
            xg = np.arange(np_sub) - cntrd
            yg = np.arange(np_sub) - cntrd
            xx0, yy0 = np.meshgrid(xg, yg)
            sigma = 1.5 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            ccd = np.zeros((np_sub, np_sub), dtype=np.float32)
            for flux, shift_dx in offsets_and_fluxes:
                xx = xx0 - shift_dx
                gaussian = np.exp(-(xx**2 + yy0**2) / (2 * sigma**2))
                gaussian = (gaussian / np.sum(gaussian)) * flux
                ccd += gaussian
            frame = xp.asarray(ccd)

            slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=0.0,
                                             gate_type='relative_ceiling',
                                             margin_exclude_radius_px=3.0,
                                             target_device_idx=target_device_idx)
            slopec.inputs['in_pixels'].set(pixels)
            self._run_frame(slopec, pixels, frame, t)
            return float(cpuArray(slopec.outputs['out_margin'].value)[0])

        baseline_margin = margin_for([(1000.0, 0.0)])
        inside_margin = margin_for([(1000.0, 0.0), (700.0, 1.0)])     # 1px, well inside 3.0px radius
        outside_margin = margin_for([(1000.0, 0.0), (700.0, 4.5)])    # 4.5px, clearly outside

        self.assertAlmostEqual(inside_margin, baseline_margin, delta=0.05,
            msg=f"A competing bump WELL WITHIN the exclusion radius changed the "
                f"margin more than expected (baseline={baseline_margin}, "
                f"inside={inside_margin}) -- it should be treated as the winning "
                f"peak's own shoulder")
        self.assertLess(outside_margin, baseline_margin - 0.3,
            f"A competing bump OUTSIDE the exclusion radius did not reduce the "
            f"margin (baseline={baseline_margin}, outside={outside_margin}) -- "
            f"it should count as a genuine competitor")
        self.assertGreater(inside_margin, outside_margin + 0.3,
            f"Expected a clearly higher margin when the competing bump is "
            f"excluded than when it is counted (inside={inside_margin}, "
            f"outside={outside_margin})")

    @cpu_and_gpu
    def test_relative_gate_telemetry_outputs_wired_correctly(self, target_device_idx, xp):
        """
        out_margin/out_rho_sq_ceiling (2026-09-17) must be registered in
        output_names() and in self.outputs as BaseValue instances, one value
        per sub-aperture, and their .value must match the internal
        margin_out/rho_sq_ceiling_out buffers after a frame -- same
        registration/verification pattern as
        test_x_c_y_c_outputs_exist_with_correct_type_and_shape /
        test_x_c_y_c_populated_by_trigger_code.
        """
        subap_npx, t = 8, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx, n_sub_side=2)
        n_subaps = subapdata.n_subaps
        self.assertEqual(n_subaps, 4)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        output_desc = AdaptiveShrinkageSlopec.output_names()
        self.assertIn('out_margin', output_desc)
        self.assertIn('out_rho_sq_ceiling', output_desc)
        self.assertIs(output_desc['out_margin'].type, BaseValue)
        self.assertIs(output_desc['out_rho_sq_ceiling'].type, BaseValue)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         gate_type='relative_ceiling',
                                         target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        for name in ('out_margin', 'out_rho_sq_ceiling'):
            self.assertIn(name, slopec.outputs, f"{name} missing from self.outputs")
            self.assertIsInstance(slopec.outputs[name], BaseValue,
                                  f"{name} is not a BaseValue instance")
            self.assertEqual(slopec.outputs[name].value.shape, (n_subaps,),
                             f"{name} does not have one value per sub-aperture")

        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=1e4, bg=0.0)
        self._run_frame(slopec, pixels, frame, t)

        margin_value = cpuArray(slopec.outputs['out_margin'].value)
        ceiling_value = cpuArray(slopec.outputs['out_rho_sq_ceiling'].value)
        np.testing.assert_array_equal(margin_value, cpuArray(slopec.margin_out),
            err_msg="out_margin.value does not match the internal margin_out buffer")
        np.testing.assert_array_equal(ceiling_value, cpuArray(slopec.rho_sq_ceiling_out),
            err_msg="out_rho_sq_ceiling.value does not match the internal "
                    "rho_sq_ceiling_out buffer")
        np.testing.assert_array_equal(ceiling_value, cpuArray(slopec._gate.rho_sq_ceiling),
            err_msg="out_rho_sq_ceiling.value does not match the actual "
                    "persistent rho_sq_ceiling state on the gate object")

        self.assertEqual(slopec.outputs['out_margin'].generation_time, t)
        self.assertEqual(slopec.outputs['out_rho_sq_ceiling'].generation_time, t)

    @cpu_and_gpu
    def test_relative_gate_no_exceptions_or_nan_across_full_flux_sweep(self, target_device_idx, xp):
        """
        Same flux sweep (including a literal all-zero frame) and structure
        as test_no_exceptions_or_nan_across_full_flux_sweep_including_exact_zero,
        but with gate_type='relative_ceiling' -- catches numerical edge cases
        (e.g. xp.inf/xp.where broadcasting or int32/float32 handling under
        cupy) that the handoff's manual spot-checks (5 flux levels only)
        might have missed. Additionally checks margin/rho_sq_ceiling stay
        finite and the ceiling never goes negative throughout.
        """
        np.random.seed(1234)
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(subapdata, fwhm_pix=1.5, ron_e=1.0,
                                         gate_type='relative_ceiling',
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
                self.fail(f"AdaptiveShrinkageSlopec (gate_type='relative_ceiling') "
                          f"raised at flux={flux}: {e!r}")

            xslopes = cpuArray(slopec.outputs['out_slopes'].xslopes)
            yslopes = cpuArray(slopec.outputs['out_slopes'].yslopes)
            w_out = cpuArray(slopec.w_out)
            margin = cpuArray(slopec.outputs['out_margin'].value)
            ceiling = cpuArray(slopec.outputs['out_rho_sq_ceiling'].value)

            self.assertTrue(np.all(np.isfinite(xslopes)), f"NaN/Inf in xslopes at flux={flux}")
            self.assertTrue(np.all(np.isfinite(yslopes)), f"NaN/Inf in yslopes at flux={flux}")
            self.assertTrue(np.all(np.isfinite(w_out)), f"NaN/Inf in w_out at flux={flux}")
            self.assertTrue(np.all(np.isfinite(margin)), f"NaN/Inf in margin at flux={flux}")
            self.assertTrue(np.all(np.isfinite(ceiling)), f"NaN/Inf in rho_sq_ceiling at flux={flux}")
            self.assertTrue(np.all(w_out >= -1e-9) and np.all(w_out <= 1.0 + 1e-9),
                            f"w_out out of [0, 1] at flux={flux}: {w_out}")
            self.assertTrue(np.all(ceiling >= -1e-9), f"rho_sq_ceiling went negative at flux={flux}: {ceiling}")

        # The literal all-zero (no background either) frame is the strictest case.
        zero_frame = xp.zeros(ccd_shape, dtype=xp.float32)
        self._run_frame(slopec, pixels, zero_frame, t * (len(flux_sweep) + 1))
        xslopes = cpuArray(slopec.outputs['out_slopes'].xslopes)
        yslopes = cpuArray(slopec.outputs['out_slopes'].yslopes)
        w_out = cpuArray(slopec.w_out)
        margin = cpuArray(slopec.outputs['out_margin'].value)
        self.assertTrue(np.all(np.isfinite(xslopes)) and np.all(np.isfinite(yslopes)),
                        "NaN/Inf on a literal all-zero frame")
        self.assertTrue(np.all(np.isfinite(w_out)), "NaN/Inf in w_out on a literal all-zero frame")
        np.testing.assert_array_equal(margin, np.zeros_like(margin),
            err_msg="margin was not exactly 0 on a literal all-zero frame")

    # =====================================================================
    # gate_type='shifted_sigmoid' (2026-09-18): end-to-end wiring test for
    # the third gate strategy, previously exercised only in isolation in
    # test_confidence_gates.py. Uses the toy-validated candidate config
    # from ShiftedSigmoidGate's docstring / RESULTS.md's "Confidence-gate
    # redesign" section: boost_mult=6.0, beta_snr=0.5, margin_thresh=0.25,
    # beta_margin=2.0.
    # =====================================================================

    @cpu_and_gpu
    def test_shifted_sigmoid_gate_no_exceptions_and_bounded_output(self, target_device_idx, xp):
        """
        Wired through the real class across a full flux sweep (same
        structure as test_relative_gate_no_exceptions_or_nan_across_full_flux_sweep):
        no NaN/exceptions, out_margin actually populated (non-zero for at
        least one frame -- confirms needs_margin=True is engaged, not
        silently left at its zero-initialized default), and out_w_smooth
        stays inside [0, 1] throughout.
        """
        np.random.seed(1234)
        subap_npx, t = 16, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveShrinkageSlopec(
            subapdata, fwhm_pix=1.5, ron_e=1.0,
            gate_type='shifted_sigmoid',
            gate_params={'boost_mult': 6.0, 'beta_snr': 0.5,
                         'margin_thresh': 0.25, 'beta_margin': 2.0},
            target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        flux_sweep = [1e6, 1e4, 1e2, 10.0, 1.0, 0.1, 0.0, 0.0]
        any_nonzero_margin = False
        for i, flux in enumerate(flux_sweep, start=1):
            noise_std = 0.5 if flux > 0 else 0.0
            frame = self.generate_spots(ccd_shape, subapdata, xp, flux=flux, bg=1.0,
                                        noise_std=noise_std)
            try:
                self._run_frame(slopec, pixels, frame, t * i)
            except Exception as e:  # pragma: no cover - failure path
                self.fail(f"AdaptiveShrinkageSlopec (gate_type='shifted_sigmoid') "
                          f"raised at flux={flux}: {e!r}")

            xslopes = cpuArray(slopec.outputs['out_slopes'].xslopes)
            yslopes = cpuArray(slopec.outputs['out_slopes'].yslopes)
            w_out = cpuArray(slopec.w_out)
            w_smooth = cpuArray(slopec.outputs['out_w_smooth'].value)
            margin = cpuArray(slopec.outputs['out_margin'].value)

            self.assertTrue(np.all(np.isfinite(xslopes)), f"NaN/Inf in xslopes at flux={flux}")
            self.assertTrue(np.all(np.isfinite(yslopes)), f"NaN/Inf in yslopes at flux={flux}")
            self.assertTrue(np.all(np.isfinite(w_out)), f"NaN/Inf in w_out at flux={flux}")
            self.assertTrue(np.all(np.isfinite(margin)), f"NaN/Inf in margin at flux={flux}")
            self.assertTrue(np.all(w_smooth >= -1e-9) and np.all(w_smooth <= 1.0 + 1e-9),
                            f"out_w_smooth out of [0, 1] at flux={flux}: {w_smooth}")
            if np.any(margin != 0.0):
                any_nonzero_margin = True

        self.assertTrue(any_nonzero_margin,
            "out_margin stayed identically 0 across the whole sweep -- "
            "ShiftedSigmoidGate.needs_margin does not appear to be engaged")

        # The literal all-zero frame is the strictest case.
        zero_frame = xp.zeros(ccd_shape, dtype=xp.float32)
        self._run_frame(slopec, pixels, zero_frame, t * (len(flux_sweep) + 1))
        xslopes = cpuArray(slopec.outputs['out_slopes'].xslopes)
        yslopes = cpuArray(slopec.outputs['out_slopes'].yslopes)
        w_smooth = cpuArray(slopec.outputs['out_w_smooth'].value)
        self.assertTrue(np.all(np.isfinite(xslopes)) and np.all(np.isfinite(yslopes)),
                        "NaN/Inf on a literal all-zero frame")
        self.assertTrue(np.all(w_smooth >= -1e-9) and np.all(w_smooth <= 1.0 + 1e-9),
                        "out_w_smooth out of [0, 1] on a literal all-zero frame")

    @cpu_and_gpu
    def test_shifted_sigmoid_gate_differs_from_wiener_at_low_snr(self, target_device_idx, xp):
        """
        Sanity check that gate_type='shifted_sigmoid' is actually wired
        into Step 3, not silently falling back to WienerGate: for the same
        low-flux frame (same k_wiener, no EMA lag), the shifted-sigmoid
        gate's w_out must differ measurably from the plain Wiener default's,
        since the two gates implement genuinely different formulas
        (min of two shifted/scaled sigmoids vs. a fixed-threshold
        rational function).
        """
        subap_npx, t = 16, int(1e9)
        subapdata_wiener, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        subapdata_sigmoid, _ = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels_wiener = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        pixels_sigmoid = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        common = dict(fwhm_pix=1.5, k_wiener=10.0, ron_e=1.0, w_ema_alpha=1.0,
                     target_device_idx=target_device_idx)
        slopec_wiener = AdaptiveShrinkageSlopec(subapdata_wiener, **common)
        slopec_sigmoid = AdaptiveShrinkageSlopec(
            subapdata_sigmoid,
            gate_type='shifted_sigmoid',
            gate_params={'boost_mult': 6.0, 'beta_snr': 0.5,
                         'margin_thresh': 0.25, 'beta_margin': 2.0},
            **common)
        slopec_wiener.inputs['in_pixels'].set(pixels_wiener)
        slopec_sigmoid.inputs['in_pixels'].set(pixels_sigmoid)

        # A single, clean, low-flux spot: low-ish SNR (rho_sq well below
        # boost_mult*k_wiener=60) but a clean, well-defined coarse peak
        # (moderate-to-high margin) -- exactly the regime where the two
        # gates' differing formulas should diverge.
        low_flux_frame = self.generate_spots(ccd_shape, subapdata_wiener, xp,
                                             flux=30.0, bg=0.0)
        self._run_frame(slopec_wiener, pixels_wiener, low_flux_frame, t)
        self._run_frame(slopec_sigmoid, pixels_sigmoid, low_flux_frame, t)

        rho_sq_wiener = float(cpuArray(slopec_wiener.outputs['out_rho_sq'].value)[0])
        rho_sq_sigmoid = float(cpuArray(slopec_sigmoid.outputs['out_rho_sq'].value)[0])
        np.testing.assert_allclose(rho_sq_sigmoid, rho_sq_wiener, rtol=1e-6,
            err_msg="rho_sq differs between the two instances on an identical "
                    "frame -- rho_sq itself must not depend on gate_type")

        w_wiener = float(cpuArray(slopec_wiener.w_out)[0])
        w_sigmoid = float(cpuArray(slopec_sigmoid.w_out)[0])

        self.assertGreater(abs(w_sigmoid - w_wiener), 1e-3,
            f"gate_type='shifted_sigmoid' gave a w_out ({w_sigmoid}) "
            f"indistinguishable from the plain Wiener default ({w_wiener}) -- "
            f"the strategy does not appear to be actually wired in")


if __name__ == '__main__':
    unittest.main()
