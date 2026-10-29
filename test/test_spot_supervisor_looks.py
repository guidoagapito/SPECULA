import specula
specula.init(0)  # Default target device

import unittest

from specula import np

from specula.processing_objects.spot_supervisor import SpotSupervisor

N = 64
RADIUS = 2.0


def spot(sx, sy, amp=300.0, sigma=1.5):
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


class TestSpotSupervisorLooks(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(5)

    def test_default_is_a_single_frame_look(self):
        sup = make()
        self.assertEqual(sup.look_frames, 1)

    def test_estimate_only_at_the_end_of_each_look(self):
        sup = make(look_frames=3, n_cons=5)                          # no move within the 9 frames
        for k in range(1, 10):
            info = sup.process_frame(spot(10.0, 5.0) + 3.0 * self.rng.standard_normal((N, N)), np.zeros(2))
            if k % 3:
                self.assertTrue(np.isnan(info['est']).all(), f'frame {k}')
            else:
                self.assertAlmostEqual(info['est'][0], 10.0, delta=1.0)
                self.assertAlmostEqual(info['est'][1], 5.0, delta=1.0)

    def test_consensus_needs_n_cons_looks(self):
        sup = make(look_frames=2)
        for k in range(1, 2 * sup.n_cons):
            sup.process_frame(spot(-9.0, 6.0) + 3.0 * self.rng.standard_normal((N, N)), np.zeros(2))
            self.assertEqual(sup.n_moves, 0, f'frame {k}')
        sup.process_frame(spot(-9.0, 6.0) + 3.0 * self.rng.standard_normal((N, N)), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.w, [-9.0, 6.0], atol=1.0)
        for _ in range(sup.k_hold + 1):                               # hold, then the feedforward
            sup.process_frame(spot(-9.0, 6.0) + 3.0 * self.rng.standard_normal((N, N)), np.zeros(2))
        np.testing.assert_allclose(sup.u_ff, [-9.0, 6.0], atol=1.0)

    def test_looks_are_registered_with_the_command(self):
        """Spot fixed in disturbance coordinates, own command ramping inside each look: one clean consensus."""
        sup = make(look_frames=2)
        d = np.array([11.0, -4.0])
        step = np.array([1.5, 0.5])
        for k in range(2 * sup.n_cons):
            u = step * k                                              # command passed at step k ...
            applied = step * (k - 1) if k >= 1 else np.zeros(2)       # ... reaches the pixels at step k + 1 (L = 1)
            sup.process_frame(spot(*(d - applied)) + 3.0 * self.rng.standard_normal((N, N)), u)
        self.assertEqual(sup.n_moves, 1)
        u_last = step * (2 * sup.n_cons - 2)
        np.testing.assert_allclose(sup.w, d - u_last, atol=1.5)

    def test_averaging_finds_a_spot_that_single_frames_miss(self):
        pos = np.array([12.0, -7.0])
        rates = {}
        for m in (1, 8):
            sup = make(look_frames=m)
            good = tot = 0
            for _ in range(24 * m):
                info = sup.process_frame(spot(*pos, amp=3.0) + 3.0 * self.rng.standard_normal((N, N)), np.zeros(2))
                if not np.isnan(info['est']).any():
                    tot += 1
                    good += int(np.linalg.norm(info['est'] - pos) < 3.0)
            rates[m] = good / tot
        self.assertGreater(rates[8], rates[1] + 0.3, rates)

    def test_dropout_resets_a_partial_look(self):
        sup = make(look_frames=4, presence=True, z_thr=6.0, z_thr_local=6.0, flux_thr=200.0, block_frames=1, k_absent=3, k_present=2)
        for _ in range(6):
            sup.process_frame(spot(0.0, 0.0) + 3.0 * self.rng.standard_normal((N, N)), np.zeros(2))
        for _ in range(4):
            sup.process_frame(3.0 * self.rng.standard_normal((N, N)), np.zeros(2))
        self.assertTrue(sup.dropout)
        self.assertEqual(sup.look_count, 0)


if __name__ == '__main__':
    unittest.main()
