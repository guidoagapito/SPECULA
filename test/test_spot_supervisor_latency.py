import specula
specula.init(0)  # Default target device

import unittest

from specula import np

from specula.processing_objects.spot_supervisor import SpotSupervisor

N = 64
RADIUS = 2.0
NOISE = 1.0


def spot(sx, sy, amp=300.0, sigma=1.5):
    c = (N - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(N) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(N) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def make(**kw):
    base = dict(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=-1, ff_mode='hold', n_cons=2, k_hold=1)
    base.update(kw)
    return SpotSupervisor(**base)


class TestCmdLatency(unittest.TestCase):
    """cmd_latency = L: in_command is the issued command, on the mirror L frames later (feedforward too)."""

    def setUp(self):
        self.rng = np.random.default_rng(3)

    def run_plant(self, sup, n, d_true, cmd_fn, latency):
        """Spot fixed at d_true in disturbance coordinates; the mirror shows the command issued `latency`
        steps earlier plus the feedforward issued `latency` steps earlier (a pure-delay plant)."""
        issued_cmd, issued_ff, log = [], [], []
        for j in range(n):
            issued_cmd.append(cmd_fn(j))
            k = j - latency
            on_mirror = (issued_cmd[k] if k >= 0 else np.zeros(2)) + (issued_ff[k - 1] if k >= 1 else np.zeros(2))
            sup.process_frame(spot(*(d_true - on_mirror)) + NOISE * self.rng.standard_normal((N, N)), issued_cmd[j])
            issued_ff.append(sup.u_ff.copy())
            log.append((sup.u_ff.copy(), sup.w.copy()))
        return log

    def test_invalid_latency(self):
        with self.assertRaises(ValueError):
            make(cmd_latency=0)

    def test_registration_uses_the_delayed_command(self):
        """Moving command (1 px/frame): registered estimates are constant only with the right latency."""
        ramp = lambda j: np.array([1.0 * j, 0.0])
        sup = make(cmd_latency=3, n_cons=100)          # never moves: only the registered estimates matter
        est = []
        orig = sup._look

        def spy(spectrum, corr, u):
            out = orig(spectrum, corr, u)
            if out is not None:
                est.append(sup._peak(out[0]) + out[1])
            return out
        sup._look = spy
        self.run_plant(sup, 12, np.array([15.0, -5.0]), ramp, latency=3)
        est = np.array(est[4:])
        np.testing.assert_allclose(est, np.tile([15.0, -5.0], (len(est), 1)), atol=1.0)

    def test_window_returns_when_the_feedforward_lands(self):
        L = 3
        sup = make(cmd_latency=L)
        log = self.run_plant(sup, 12, np.array([12.0, 8.0]), lambda j: np.zeros(2), latency=L)
        k_ff = next(i for i, (uff, _) in enumerate(log) if np.any(uff != 0))
        # the window stays at the hold position for L-1 steps after the feedforward is issued
        for i in range(k_ff, k_ff + L - 1):
            self.assertTrue(np.any(log[i][1] != 0), f'window returned early at step {i}')
        np.testing.assert_array_equal(log[k_ff + L - 1][1], [0.0, 0.0])

    def test_matched_latency_ends_captured_and_single_move(self):
        L = 2
        sup = make(cmd_latency=L)
        self.run_plant(sup, 20, np.array([12.0, 8.0]), lambda j: np.zeros(2), latency=L)
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.u_ff, [12.0, 8.0], atol=1.0)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])

    def test_no_new_move_while_return_in_flight(self):
        sup = make(cmd_latency=4)
        self.run_plant(sup, 7, np.array([12.0, 8.0]), lambda j: np.zeros(2), latency=4)
        self.assertTrue(sup.w_queue or sup.n_moves == 1)
        moves = sup.n_moves
        while sup.w_queue:
            sup.process_frame(spot(0.0, 0.0), np.zeros(2))
            self.assertEqual(sup.n_moves, moves)

    def test_n3_window_follows_each_feedforward_step(self):
        L = 2
        sup = make(cmd_latency=L, ff_mode='n3')
        log = self.run_plant(sup, 20, np.array([9.0, -6.0]), lambda j: np.zeros(2), latency=L)
        steps = [i for i in range(1, len(log)) if np.any(log[i][0] != log[i - 1][0])]
        self.assertEqual(len(steps), 3)
        for i in steps:
            # the window changes L-1 steps after each feedforward increment, never before
            self.assertTrue(np.array_equal(log[i][1], log[i - 1][1]))
            self.assertFalse(np.array_equal(log[i + L - 1][1], log[i + L - 2][1]))
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])

    def test_latency_one_matches_immediate_return(self):
        """L = 1: the feedforward lands on the next frame, like the window: no queueing."""
        sup = make(cmd_latency=1)
        log = self.run_plant(sup, 10, np.array([12.0, 8.0]), lambda j: np.zeros(2), latency=1)
        k_ff = next(i for i, (uff, _) in enumerate(log) if np.any(uff != 0))
        np.testing.assert_array_equal(log[k_ff][1], [0.0, 0.0])


class TestFarOnLatestLook(unittest.TestCase):
    """A captured spot must not trigger a move, even if the command drifts over the consensus window."""

    def test_command_drift_alone_does_not_move_a_captured_spot(self):
        # spot fixed on the detector at the window while the command follows a drifting disturbance: the
        # registered positions drift by 0.6 px/frame, so the median of the looks lags the current command by
        # > delta. The guard is disabled (q_thr > 1) so that only the "far" test decides.
        rng = np.random.default_rng(5)
        sup = make(n_cons=2, look_frames=4, q_thr=1.1)
        for j in range(40):
            sup.process_frame(spot(0.0, 0.0) + NOISE * rng.standard_normal((N, N)), np.array([0.6 * j, 0.0]))
        self.assertEqual(sup.n_moves, 0)

    def test_window_return_clears_the_looks(self):
        sup = make(cmd_latency=3)
        issued = []
        rng = np.random.default_rng(6)
        for j in range(12):
            on_mirror = issued[j - 4] if j >= 4 else np.zeros(2)
            sup.process_frame(spot(*(np.array([12.0, 8.0]) - on_mirror)) + NOISE * rng.standard_normal((N, N)),
                              np.zeros(2))
            issued.append(sup.u_ff.copy())
            if np.any(sup.u_ff != 0) and not sup.w_queue and np.all(sup.w == 0):
                self.assertEqual(sup.hist, [])
                break
        else:
            self.fail('the window never returned')


if __name__ == '__main__':
    unittest.main()
