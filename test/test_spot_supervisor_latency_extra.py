import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
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


def trigger_once(sup, frame, cmd, t):
    """Push one frame/command pair through the real SPECULA input/output machinery (trigger_code)."""
    pixels = Pixels(N, N, target_device_idx=-1)
    pixels.pixels = frame
    pixels.generation_time = t
    c = BaseValue(value=np.asarray(cmd, dtype=float), target_device_idx=-1)
    c.generation_time = t
    sup.inputs['in_pixels'].set(pixels)
    sup.inputs['in_command'].set(c)
    sup.check_ready(t)
    sup.trigger()
    sup.post_trigger()


class TestCmdLatencyNoneEquivalence(unittest.TestCase):
    """cmd_latency=None must be byte-for-byte identical to not passing the argument at all."""

    def test_none_matches_the_legacy_default_on_a_scenario_with_a_move(self):
        rng = np.random.default_rng(9)
        pos = np.array([11.0, -6.0])
        frames = [spot(*pos) + NOISE * rng.standard_normal((N, N)) for _ in range(20)]
        cmd = np.zeros(2)
        sup_default = make()
        sup_none = make(cmd_latency=None)
        self.assertIsNone(sup_default.cmd_latency)
        self.assertIsNone(sup_none.cmd_latency)
        for i, frame in enumerate(frames):
            info_d = sup_default.process_frame(frame.copy(), cmd)
            info_n = sup_none.process_frame(frame.copy(), cmd)
            np.testing.assert_array_equal(sup_default.w, sup_none.w, err_msg=f'w mismatch at frame {i}')
            np.testing.assert_array_equal(sup_default.u_ff, sup_none.u_ff, err_msg=f'u_ff mismatch at frame {i}')
            np.testing.assert_array_equal(info_d['est'], info_n['est'], err_msg=f'est mismatch at frame {i}')
            self.assertEqual(info_d['ratio'], info_n['ratio'])
        self.assertEqual(sup_default.n_moves, sup_none.n_moves)
        self.assertGreater(sup_default.n_moves, 0)          # make sure the scenario actually exercised a move


class TestDropoutDuringFlight(unittest.TestCase):
    """A presence dropout can start while a delayed return is queued; the mirror must still get the
    feedforward regardless (dropout only zeroes the slopes downstream, it does not touch the queue)."""

    def setUp(self):
        self.rng = np.random.default_rng(3)

    def noise(self):
        return NOISE * self.rng.standard_normal((N, N))

    def test_dropout_starting_mid_flight_does_not_cancel_the_queued_return(self):
        L = 6
        sup = make(cmd_latency=L, presence=True, z_thr=6.0, k_absent=1, k_present=2, block_frames=1)
        pos = np.array([12.0, 8.0])
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        guard = 0
        while not sup.w_queue and guard < 30:
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
            guard += 1
        self.assertTrue(sup.w_queue, 'scenario did not reach the queued-return phase')
        self.assertFalse(sup.dropout)
        u_ff_held = sup.u_ff.copy()
        self.assertTrue(np.any(u_ff_held != 0))
        # k_absent=1: a single dark block is already enough to declare a dropout, while the
        # return is still in flight (queued several frames ahead)
        sup.process_frame(self.noise(), np.zeros(2))
        self.assertTrue(sup.dropout)
        self.assertTrue(sup.w_queue, 'the return landed too early for this test to be meaningful')
        np.testing.assert_array_equal(sup.u_ff, u_ff_held)
        guard = 0
        while sup.w_queue and guard < 30:
            sup.process_frame(self.noise(), np.zeros(2))
            guard += 1
        self.assertTrue(sup.dropout)                        # still blind: no presence recovery was fed
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])     # the queued return was applied anyway
        np.testing.assert_array_equal(sup.u_ff, u_ff_held)
        self.assertEqual(sup.n_aborts, 0)


class TestConfirmWithLatency(unittest.TestCase):
    """confirm=True gates the feedforward on evidence at the new window; with cmd_latency the
    feedforward (and its window return) additionally waits for the transport delay."""

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def noise(self):
        return NOISE * self.rng.standard_normal((N, N))

    def make_conf(self, **kw):
        base = dict(cmd_latency=4, ff_mode='hold', confirm=True, z_thr_local=6.0, flux_thr=200.0,
                    block_frames=1, confirm_frames=3, k_hold=3, n_cons=5, q_thr=1.1)
        base.update(kw)
        return make(**base)

    def test_aborted_move_restores_w_prev_immediately_without_queueing(self):
        """Nothing at the new window: the move must be undone at once (no feedforward was ever
        issued, so there is nothing to wait for on the mirror side)."""
        sup = self.make_conf()
        pos = (12.0, -9.0)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        w_prev = sup.w_prev.copy()
        for _ in range(sup.k_hold + sup.confirm_frames + 3):
            sup.process_frame(self.noise(), np.zeros(2))
            self.assertEqual(sup.w_queue, [])                # abort never goes through the queue
        self.assertFalse(sup.confirmed)
        self.assertEqual(sup.n_aborts, 1)
        np.testing.assert_array_equal(sup.w, w_prev)
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])
        self.assertIsNone(sup.hold_until)

    def test_confirmed_move_queues_the_return(self):
        sup = self.make_conf()
        pos = (12.0, -9.0)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        w_held = sup.w.copy()
        guard = 0
        while sup.confirmed is not True and guard < 30:
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
            guard += 1
        self.assertTrue(sup.confirmed)
        np.testing.assert_allclose(sup.u_ff, w_held)          # feedforward issued as soon as confirmed
        self.assertTrue(sup.w_queue, 'the return should be queued, not applied at once (cmd_latency > 1)')
        np.testing.assert_array_equal(sup.w, w_held)          # window still at the hold position, in flight
        guard = 0
        while sup.w_queue and guard < 30:
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
            guard += 1
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        self.assertEqual(sup.n_aborts, 0)


class TestResetStateClearsLatencyState(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_reset_state_clears_a_pending_return_and_histories(self):
        L = 4
        sup = make(cmd_latency=L)
        pos = np.array([12.0, 8.0])
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + NOISE * self.rng.standard_normal((N, N)), np.zeros(2))
        guard = 0
        while not sup.w_queue and guard < 30:
            sup.process_frame(spot(*pos) + NOISE * self.rng.standard_normal((N, N)), np.zeros(2))
            guard += 1
        self.assertTrue(sup.w_queue)
        self.assertGreater(len(sup.cmd_hist), 0)
        self.assertGreater(len(sup.uff_hist), 0)
        sup.reset_state()
        self.assertEqual(sup.w_queue, [])
        self.assertEqual(len(sup.cmd_hist), 0)
        self.assertEqual(len(sup.uff_hist), 0)
        self.assertEqual(sup.cmd_hist.maxlen, L + 1)
        self.assertEqual(sup.uff_hist.maxlen, L)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])
        self.assertEqual(sup.frame, 0)
        # and it works again from a clean slate (n_moves is a lifetime counter, not reset here,
        # matching test_reset_state_clears_dynamic_state in test_spot_supervisor_extra.py)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + NOISE * self.rng.standard_normal((N, N)), np.zeros(2))
        self.assertEqual(sup.n_moves, 2)


class TestStartupRegistration(unittest.TestCase):
    """Before L+1 commands have been seen, ``_mirror_command`` falls back to a zero command."""

    def setUp(self):
        self.rng = np.random.default_rng(5)

    def test_mirror_command_falls_back_to_zero_before_the_history_fills(self):
        L = 3
        sup = make(cmd_latency=L)
        cmds = [np.array([10.0 * i, -5.0 * i]) for i in range(1, L + 3)]
        outs = [sup._mirror_command(c) for c in cmds]
        for k in range(L):
            np.testing.assert_array_equal(outs[k], [0.0, 0.0], err_msg=f'call {k} should still be zero')
        # the (L+1)-th call is the first one to see a delayed value: the oldest queued command
        np.testing.assert_array_equal(outs[L], cmds[0])

    def test_startup_with_zero_command_and_spot_at_reference_is_a_no_op(self):
        """No crash during the pre-fill frames, and no spurious move: a zero command registered
        as zero (real or fallback, here they coincide) with the spot exactly at the reference."""
        L = 4
        sup = make(cmd_latency=L)
        for _ in range(L + 3):
            info = sup.process_frame(spot(0.0, 0.0) + NOISE * self.rng.standard_normal((N, N)), np.zeros(2))
            self.assertTrue(np.all(np.isfinite(info['est'])))
        self.assertEqual(sup.n_moves, 0)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])


class TestTelemetryDuringFlight(unittest.TestCase):
    """out_window / out_feedforward (via trigger_code) must keep reporting the hold window and the
    already-issued feedforward for as long as the return is in flight."""

    def setUp(self):
        self.rng = np.random.default_rng(4)

    def noise(self):
        return NOISE * self.rng.standard_normal((N, N))

    def test_telemetry_reflects_the_hold_window_during_the_flight(self):
        L = 3
        sup = make(cmd_latency=L)
        pos = np.array([12.0, 8.0])
        t = 1
        for _ in range(sup.n_cons):
            trigger_once(sup, spot(*pos) + self.noise(), [0.0, 0.0], t)
            t += 1
        guard = 0
        while not sup.w_queue and guard < 30:
            trigger_once(sup, spot(*pos) + self.noise(), [0.0, 0.0], t)
            t += 1
            guard += 1
        self.assertTrue(sup.w_queue, 'scenario did not reach the queued-return phase')
        ff_at_hold = cpuArray(sup.outputs['out_feedforward'].value).copy()
        self.assertTrue(np.any(ff_at_hold != 0))
        guard = 0
        while sup.w_queue and guard < 30:
            win = cpuArray(sup.outputs['out_window'].value)
            np.testing.assert_allclose(win[:2], pos, atol=1.0)           # still the hold window
            self.assertEqual(win[2], 0.0)                                # not the dropout hold flag
            np.testing.assert_array_equal(cpuArray(sup.outputs['out_feedforward'].value), ff_at_hold)
            trigger_once(sup, spot(*pos) + self.noise(), [0.0, 0.0], t)
            t += 1
            guard += 1
        win = cpuArray(sup.outputs['out_window'].value)
        np.testing.assert_array_equal(win[:2], [0.0, 0.0])
        self.assertEqual(sup.n_moves, 1)                                 # no spurious second move meanwhile


if __name__ == '__main__':
    unittest.main()
