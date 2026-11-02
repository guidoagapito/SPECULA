import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.processing_objects.spot_supervisor import SpotSupervisor, NM_TO_PX
from test.specula_testlib import cpu_and_gpu

N = 64
RADIUS = 2.0
SIGMA_W = 2.0 * RADIUS / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def spot(sx, sy, amp=300.0, sigma=1.5):
    """Gaussian spot at (sx, sy) px from the frame centre (x = column, y = row)."""
    c = (N - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(N) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(N) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def make(target_device_idx=-1, **kw):
    """Presence off and confirmation thresholds that always pass: tests of the other mechanisms."""
    kw.setdefault('presence', False)
    kw.setdefault('cmd_latency', 1)
    kw.setdefault('z_thr_local', -np.inf)
    kw.setdefault('flux_thr', -np.inf)
    return SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=target_device_idx, **kw)


def feed(sup, rng, position, n_frames, u=(0.0, 0.0), noise=3.0, amp=300.0):
    """Feed frames with the spot at ``position`` (detector px); returns the last telemetry."""
    info = None
    for _ in range(n_frames):
        frame = spot(*position, amp=amp) + noise * rng.standard_normal((N, N))
        info = sup.process_frame(frame, np.asarray(u, float))
    return info


class TestSpotSupervisor(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(3)

    @cpu_and_gpu
    def test_estimate_and_axis_convention(self, target_device_idx, xp):
        sup = make(target_device_idx)
        info = feed(sup, self.rng, (12.0, -9.0), 1)
        self.assertAlmostEqual(info['est'][0], 12.0, delta=1.0)      # x = column
        self.assertAlmostEqual(info['est'][1], -9.0, delta=1.0)      # y = row

    def test_no_move_while_the_spot_is_in_the_window(self):
        sup = make()
        feed(sup, self.rng, (0.5, -0.5), 30)
        self.assertEqual(sup.n_moves, 0)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])

    def test_hold_moves_window_then_feedforward_in_one_shot(self):
        sup = make()
        pos = (12.0, -9.0)
        feed(sup, self.rng, pos, sup.n_cons)                          # consensus reached on the last frame
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.w, pos, atol=1.0)              # window on the found spot
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])           # not moved yet: hold phase
        w_held = sup.w.copy()
        feed(sup, self.rng, pos, sup.k_hold + 1)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])              # window back to the reference
        np.testing.assert_allclose(sup.u_ff, w_held)                  # feedforward == held window centre
        self.assertEqual(sup.n_moves, 1)

    def test_registration_with_a_moving_command(self):
        """The spot is fixed in disturbance coordinates while our own command steps: still one consensus."""
        sup = make()
        d = np.array([10.0, 4.0])
        u = np.array([3.0, -2.0])                                     # correction px, spot shift
        # cmd_latency = 1: the command passed at step j reaches the pixels at step j + 1
        for k in range(sup.n_cons):
            applied = u if k >= 1 else np.zeros(2)
            sup.process_frame(spot(*(d - applied)) + 3.0 * self.rng.standard_normal((N, N)), u)
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.w, d - u, atol=1.0)            # window on the spot as seen now
        feed(sup, self.rng, d - u, sup.k_hold + 2, u=u)
        np.testing.assert_allclose(sup.u_ff, d - u, atol=1.0)         # feedforward brings the spot to the reference

    def test_guard_c_vetoes_a_move_on_a_capturing_window(self):
        far = (15.0, 0.0)
        strong = spot(*far, amp=300.0)
        for amp_near, expect_moves in [(240.0, 0), (60.0, 1)]:
            sup = make()
            for _ in range(sup.n_cons + 2):
                frame = strong + spot(0.0, 0.0, amp=amp_near) + 3.0 * self.rng.standard_normal((N, N))
                sup.process_frame(frame, np.zeros(2))
            self.assertEqual(sup.n_moves, expect_moves, f'amp_near={amp_near}')

    def test_no_consensus_on_scattered_estimates(self):
        sup = make()
        positions = [(12, 8), (-14, 3), (5, -15), (-9, -9), (16, 12), (-3, 14)]
        for p in positions:
            sup.process_frame(spot(*p), np.zeros(2))
        self.assertEqual(sup.n_moves, 0)

    def test_presence_dropout_and_release(self):
        z_thr = 6.0
        sup = make(presence=True, z_thr=z_thr, z_thr_local=6.0, flux_thr=200.0, k_absent=4, k_present=3)
        feed(sup, self.rng, (0.0, 0.0), 10)
        self.assertFalse(sup.dropout)
        for _ in range(8):                                            # star off: noise only
            sup.process_frame(3.0 * self.rng.standard_normal((N, N)), np.zeros(2))
        self.assertTrue(sup.dropout)
        moves = sup.n_moves
        feed(sup, self.rng, (14.0, 5.0), 2)                           # star back, not yet released
        self.assertTrue(sup.dropout)
        self.assertEqual(sup.n_moves, moves)                          # nothing moves during a dropout
        feed(sup, self.rng, (14.0, 5.0), 12)
        self.assertFalse(sup.dropout)
        self.assertGreaterEqual(sup.n_moves, moves + 1)               # after release the spot is reacquired

    def test_presence_needs_threshold(self):
        with self.assertRaises(ValueError):
            make(presence=True)

    def test_local_thresholds_are_required(self):
        for missing in ('z_thr_local', 'flux_thr'):
            with self.assertRaises(ValueError, msg=missing):
                SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, cmd_latency=1, presence=False,
                               target_device_idx=-1, **{k: 1.0 for k in ('z_thr_local', 'flux_thr') if k != missing})

    def test_cmd_latency_is_required_and_positive(self):
        with self.assertRaises(TypeError):
            SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, presence=False, z_thr_local=1.0, flux_thr=1.0,
                           target_device_idx=-1)
        with self.assertRaises(ValueError):
            make(cmd_latency=0)

    def test_trigger_outputs_window_and_feedforward_in_nm(self):
        sup = make()
        pos = (11.0, -6.0)
        t = 1
        for _ in range(sup.n_cons + sup.k_hold):                      # consensus, hold, feedforward (one move)
            pixels = Pixels(N, N, target_device_idx=-1)
            pixels.pixels = spot(*pos) + 3.0 * self.rng.standard_normal((N, N))
            pixels.generation_time = t
            cmd = BaseValue(value=np.zeros(2), target_device_idx=-1)
            cmd.generation_time = t
            sup.inputs['in_pixels'].set(pixels)
            sup.inputs['in_command'].set(cmd)
            sup.check_ready(t)
            sup.trigger()
            sup.post_trigger()
            t += 1
        ff_nm = cpuArray(sup.outputs['out_feedforward'].value)
        np.testing.assert_allclose(ff_nm * NM_TO_PX, sup.u_ff, rtol=1e-5)
        np.testing.assert_allclose(ff_nm * NM_TO_PX, pos, atol=1.0)
        win = cpuArray(sup.outputs['out_window'].value)
        np.testing.assert_array_equal(win, [0.0, 0.0, 0.0])
        self.assertEqual(sup.outputs['out_state'].generation_time, t - 1)

    def test_command_matrix_is_inverted_for_the_output(self):
        m = np.array([[0.0, -0.003], [0.003, 0.0]])                   # axes swapped, one sign flipped
        sup = make(cmd_to_px=m.tolist())
        feed(sup, self.rng, (9.0, 3.0), sup.n_cons + sup.k_hold + 2)
        np.testing.assert_allclose(m @ (sup.px_to_cmd @ sup.u_ff), sup.u_ff, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
