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


if __name__ == '__main__':
    unittest.main()
