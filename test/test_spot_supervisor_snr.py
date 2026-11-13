import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.processing_objects.spot_supervisor_snr import SpotSupervisorSnr
from test.specula_testlib import cpu_and_gpu

N = 64
RADIUS = 2.0
PEDESTAL = 50.0


def spot(sx, sy, amp=100.0, sigma=1.2):
    """Gaussian spot at (sx, sy) px from the frame centre (x = column, y = row)."""
    c = (N - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(N) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(N) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def make(target_device_idx=-1, **kw):
    """Presence off and confirmation thresholds that always pass: tests of the locator and the inherited moves."""
    kw.setdefault('presence', False)
    kw.setdefault('cmd_latency', 1)
    kw.setdefault('z_thr_local', -np.inf)
    kw.setdefault('flux_thr', -np.inf)
    return SpotSupervisorSnr(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=target_device_idx, **kw)


def feed(sup, rng, spots, n_frames, noise=2.0, pedestal=PEDESTAL):
    """Feed frames made of ``spots`` = [(x, y, amp), ...] on a pedestal plus noise; returns the last telemetry."""
    info = None
    for _ in range(n_frames):
        frame = pedestal + noise * rng.standard_normal((N, N))
        for x, y, amp in spots:
            frame = frame + spot(x, y, amp=amp)
        info = sup.process_frame(frame, np.zeros(2))
    return info


class TestSpotSupervisorSnr(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(5)

    def test_move_on_the_look_boundary_then_feedforward_and_return(self):
        sup = make(look_frames=4)
        pos = (12.0, -9.0)
        feed(sup, self.rng, [(*pos, 200.0)], 3)
        self.assertEqual(sup.n_moves, 0)                              # look incomplete: no decision
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        info = feed(sup, self.rng, [(*pos, 200.0)], 1)
        self.assertEqual(sup.n_moves, 1)
        self.assertGreater(info['ratio'], sup.snr_thr)
        np.testing.assert_allclose(sup.w, pos, atol=1.0)              # window on the spot (integer peak)
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])           # hold phase
        w_held = sup.w.copy()
        feed(sup, self.rng, [(*pos, 200.0)], sup.k_hold + 1)
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])              # window back to the reference
        np.testing.assert_allclose(sup.u_ff, w_held)                  # feedforward == held window centre

    def test_axis_convention(self):
        sup = make()
        info = feed(sup, self.rng, [(-14.0, 7.0, 200.0)], 1)
        self.assertAlmostEqual(info['est'][0], -14.0, delta=1.0)      # x = column
        self.assertAlmostEqual(info['est'][1], 7.0, delta=1.0)        # y = row

    def test_no_move_while_the_spot_is_in_the_window(self):
        sup = make(look_frames=2)
        for pos in [(0.5, -0.5), (2.0, 1.0), (-1.0, 2.5)]:            # all within capture_radius = 4 px
            sup = make(look_frames=2)
            for _ in range(20):
                info = feed(sup, self.rng, [(*pos, 100.0)], 2)
                self.assertGreater(info['ratio'], sup.snr_thr, msg=str(pos))
            self.assertEqual(sup.n_moves, 0, msg=str(pos))
            np.testing.assert_array_equal(sup.w, [0.0, 0.0])
            np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])

    def test_pure_noise_never_moves(self):
        sup = make(look_frames=1)
        for _ in range(300):
            feed(sup, self.rng, [], 1, noise=3.0)
        self.assertEqual(sup.n_moves, 0)

    def test_faint_far_spot_does_not_move(self):
        sup = make(look_frames=1)
        # peak amplitude 1 against noise 3 per frame: SNR of a single-frame look is far below snr_thr
        info = feed(sup, self.rng, [(15.0, 10.0, 1.0)], 100, noise=3.0)
        self.assertEqual(sup.n_moves, 0)
        self.assertLess(info['ratio'], sup.snr_thr)

    def test_snr_threshold_scales_the_decision(self):
        """The same spot (SNR ~ 15 at 8 frames) moves the window below snr_thr and not above it."""
        pos = (15.0, -10.0, 8.0)
        sup_hi = make(look_frames=8, snr_thr=30.0)
        sup_lo = make(look_frames=8, snr_thr=7.5)
        for sup in (sup_hi, sup_lo):
            rng = np.random.default_rng(11)
            feed(sup, rng, [pos], 8, noise=3.0)
        self.assertEqual(sup_hi.n_moves, 0)
        self.assertEqual(sup_lo.n_moves, 1)

    def test_local_veto_when_the_window_already_sees_a_spot(self):
        far = (15.0, -10.0, 300.0)
        for amp_near, expect_moves in [(60.0, 0), (0.0, 1)]:
            sup = make(look_frames=1)
            feed(sup, self.rng, [(0.0, 0.0, amp_near), far], 2)   # before the return at k_hold
            self.assertEqual(sup.n_moves, expect_moves, msg=f'amp_near={amp_near}')

    def test_look_restarts_at_a_window_change(self):
        sup = make(look_frames=4, k_hold=2)
        pos = [(12.0, -9.0, 200.0)]
        feed(sup, self.rng, pos, 3)
        self.assertEqual(sup.look_count, 3)
        feed(sup, self.rng, pos, 1)
        self.assertEqual(sup.n_moves, 1)
        self.assertEqual(sup.look_count, 0)                           # restarted at the move
        feed(sup, self.rng, pos, 2)                                   # frames 5, 6: return fires at frame 6
        self.assertEqual(sup.look_count, 0)                           # restarted again at the window return
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        feed(sup, self.rng, pos, 3)
        self.assertEqual(sup.look_count, 3)
        self.assertEqual(sup.n_moves, 1)                              # no decision on a look straddling the change

    def test_look_after_a_move_holds_only_frames_after_it(self):
        """Frames before the move show the spot; after it the sky is empty: the next look must be empty."""
        sup = make(look_frames=4, k_hold=100)                         # no return, so the look is not restarted again
        feed(sup, self.rng, [(12.0, -9.0, 200.0)], 4)
        self.assertEqual(sup.n_moves, 1)
        feed(sup, self.rng, [], 4)
        self.assertLess(float(cpuArray(sup.last_look).max()), PEDESTAL + 10.0)   # no leftover spot flux

    def test_no_second_move_while_a_return_is_in_flight(self):
        sup = make(look_frames=1, cmd_latency=3, k_hold=1)
        first, other = (12.0, -9.0, 200.0), (-15.0, 12.0, 200.0)
        feed(sup, self.rng, [first], 1)                               # frame 0: move
        self.assertEqual(sup.n_moves, 1)
        feed(sup, self.rng, [first], 1)                               # frame 1: feedforward issued, return queued
        self.assertEqual(len(sup.w_queue), 1)
        np.testing.assert_allclose(sup.w, first[:2], atol=1.0)        # window still on the first spot
        feed(sup, self.rng, [other], 1)                               # frame 2: far spot, window blind, return pending
        self.assertEqual(sup.n_moves, 1)                              # blocked by the in-flight return
        feed(sup, self.rng, [other], 1)                               # frame 3: return lands
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        feed(sup, self.rng, [other], 1)                               # frame 4: free again
        self.assertEqual(sup.n_moves, 2)

    def test_presence_dropout_and_release_with_no_looks_during_dropout(self):
        sup = make(presence=True, z_thr=6.0, z_thr_local=6.0, flux_thr=200.0, k_absent=4, k_present=3,
                   look_frames=2)
        feed(sup, self.rng, [(0.0, 0.0, 300.0)], 10, pedestal=0.0, noise=3.0)
        self.assertFalse(sup.dropout)
        self.assertEqual(sup.window_value()[2], 0.0)
        for _ in range(8):                                            # star off: noise only
            feed(sup, self.rng, [], 1, pedestal=0.0, noise=3.0)
        self.assertTrue(sup.dropout)
        self.assertEqual(sup.window_value()[2], 1.0)
        self.assertEqual(sup.n_moves, 0)
        self.assertEqual(sup.look_count, 0)                           # no look accumulated while blind
        feed(sup, self.rng, [(14.0, 5.0, 300.0)], 2, pedestal=0.0, noise=3.0)   # back, not yet released
        self.assertTrue(sup.dropout)
        self.assertEqual(sup.n_moves, 0)
        self.assertEqual(sup.look_count, 0)
        feed(sup, self.rng, [(14.0, 5.0, 300.0)], 12, pedestal=0.0, noise=3.0)
        self.assertFalse(sup.dropout)
        self.assertEqual(sup.window_value()[2], 0.0)
        self.assertGreaterEqual(sup.n_moves, 1)                       # reacquired after the release

    def test_disabled_does_not_move(self):
        sup = make(look_frames=1)
        sup.set_enabled(False)
        info = feed(sup, self.rng, [(12.0, -9.0, 200.0)], 5)
        self.assertEqual(sup.n_moves, 0)
        self.assertGreater(info['ratio'], sup.snr_thr)                # still measured (passive)
        np.testing.assert_allclose(info['est'], [12.0, -9.0], atol=1.0)

    @cpu_and_gpu
    def test_trigger_outputs(self, target_device_idx, xp):
        sup = make(target_device_idx, look_frames=4, k_hold=100)      # long hold: window stays on the spot
        pos = (11.0, -6.0)
        t = 1
        frames = []
        for _ in range(sup.look_frames):
            pixels = Pixels(N, N, target_device_idx=target_device_idx)
            frame = PEDESTAL + 2.0 * self.rng.standard_normal((N, N)) + spot(*pos, amp=200.0)
            frames.append(frame)
            pixels.pixels = xp.asarray(frame, dtype=pixels.pixels.dtype)
            pixels.generation_time = t
            cmd = BaseValue(value=xp.zeros(2), target_device_idx=target_device_idx)
            cmd.generation_time = t
            sup.inputs['in_pixels'].set(pixels)
            sup.inputs['in_command'].set(cmd)
            sup.check_ready(t)
            sup.trigger()
            sup.post_trigger()
            t += 1
        win = cpuArray(sup.outputs['out_window'].value)
        np.testing.assert_allclose(win[:2], pos, atol=1.0)
        self.assertEqual(win[2], 0.0)                                 # no dropout
        state = cpuArray(sup.outputs['out_state'].value)
        self.assertGreater(state[3], sup.snr_thr)                     # SNR of the last look
        self.assertEqual(state[5], 1.0)                               # n_moves
        np.testing.assert_allclose(state[:2], pos, atol=1.0)
        look = cpuArray(sup.outputs['out_look'].value)
        np.testing.assert_allclose(look, np.mean(frames, axis=0), rtol=1e-4, atol=1e-3)
        self.assertEqual(sup.outputs['out_state'].generation_time, t - 1)


if __name__ == '__main__':
    unittest.main()
