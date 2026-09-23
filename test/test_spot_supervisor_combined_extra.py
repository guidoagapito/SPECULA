import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.processing_objects.spot_supervisor import SpotSupervisor
from test.specula_testlib import cpu_and_gpu

N = 64
RADIUS = 2.0
NOISE = 3.0


def spot(sx, sy, amp=300.0, sigma=1.5):
    """Gaussian spot at (sx, sy) px from the frame centre (x = column, y = row)."""
    c = (N - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(N) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(N) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def make(target_device_idx=-1, **kw):
    return SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=target_device_idx, **kw)


class TestConfirmationEdgeCases(unittest.TestCase):
    """Edge cases of the confirm/hold interaction not covered by test_spot_supervisor_combined.py.

    q_thr=1.1 disables guard C (a ratio in [0, 1] is always < 1.1), matching the convention
    already used in test_spot_supervisor_extra.py to isolate the consensus/confirm logic.
    """

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def noise(self):
        return NOISE * self.rng.standard_normal((N, N))

    def make_conf(self, **kw):
        base = dict(ff_mode='hold', confirm=True, z_thr_local=6.0, flux_thr=200.0,
                    block_frames=1, confirm_frames=3, k_hold=3, n_cons=5, q_thr=1.1)
        base.update(kw)
        return make(**base)

    def test_second_consensus_during_confirmation_keeps_original_w_prev_and_restarts(self):
        """A wrong consensus (transient) is immediately followed by the real one, before the first
        confirmation block completes: w_prev must still be the window active BEFORE the first move
        (not the transient's), and the confirmation window/counters must restart on the new target."""
        sup = self.make_conf(confirm_frames=20)          # long enough to still be open for the 2nd consensus
        pos1, pos2 = (12.0, -9.0), (-14.0, 10.0)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos1) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_array_equal(sup.w_prev, [0.0, 0.0])
        self.assertTrue(sup.conf_active)
        w1 = sup.w.copy()
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos2) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 2)
        np.testing.assert_allclose(sup.w, pos2, atol=1.0)
        self.assertFalse(np.allclose(sup.w, w1))
        np.testing.assert_array_equal(sup.w_prev, [0.0, 0.0])      # still the original, not w1
        self.assertTrue(sup.conf_active)                           # confirmation restarted for pos2
        self.assertEqual(sup.conf_count, 0)

    def test_dropout_during_confirmation_hold_discards_the_blind_verdict(self):
        """Presence goes absent right after a move. A confirmation verdict formed on the dark frames is
        discarded when the dropout starts, nothing is collected while blind, and after release the
        confirmation restarts on fresh frames: no spurious abort, the pending move completes."""
        sup = make(ff_mode='hold', confirm=True, presence=True, z_thr=6.0, z_thr_local=6.0,
                   flux_thr=200.0, block_frames=1, confirm_frames=3, k_hold=3, n_cons=5,
                   k_absent=3, k_present=2, q_thr=1.1)
        pos = (12.0, -9.0)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        w_held = sup.w.copy()
        for _ in range(9):                                 # long dark spell: dropout while the hold is pending
            sup.process_frame(self.noise(), np.zeros(2))
        self.assertTrue(sup.dropout)
        self.assertIsNone(sup.confirmed)                   # blind verdict discarded, nothing collected since
        self.assertEqual(sup.conf_count, 0)
        np.testing.assert_array_equal(sup.w, w_held)        # frozen: no abort/restore while blind
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])
        self.assertEqual(sup.n_aborts, 0)
        for _ in range(2):                                 # release needs k_present = 2 hits
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertFalse(sup.dropout)
        for _ in range(sup.confirm_frames + 1):             # fresh confirmation at the held window
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_aborts, 0)
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.u_ff, w_held)        # the pending move completed

    def test_confirmation_works_with_window_near_the_frame_edge(self):
        """Window move landing close to the border: the Gaussian mask / local disk are clipped by
        the frame but must not crash and must still confirm a real spot there."""
        sup = self.make_conf()
        pos = (29.0, -29.0)                                 # half-width is (N-1)/2 = 31.5
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        w1 = sup.w.copy()
        for _ in range(max(sup.k_hold, sup.confirm_frames) + 2):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertTrue(sup.confirmed)
        np.testing.assert_allclose(sup.u_ff, w1)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        self.assertEqual(sup.n_aborts, 0)

    def test_n_aborts_accumulate_over_repeated_wrong_consensus(self):
        sup = self.make_conf()
        positions = [(12.0, -9.0), (-14.0, 8.0), (10.0, 10.0)]
        for i, pos in enumerate(positions, start=1):
            for _ in range(sup.n_cons):
                sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
            for _ in range(sup.k_hold + sup.confirm_frames + 1):
                sup.process_frame(self.noise(), np.zeros(2))       # nothing at the new window every time
            self.assertEqual(sup.n_aborts, i)
            self.assertEqual(sup.n_moves, i)
            np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])    # never any spurious feedforward

    def test_abort_clears_hold_and_n3_state(self):
        """ff_mode='n3': an abort must leave no dangling n3 sequence (n3_left = 0, hold_until = None)."""
        sup = make(ff_mode='n3', confirm=True, z_thr_local=6.0, flux_thr=200.0,
                   confirm_frames=3, k_hold=2, n_cons=5, q_thr=1.1)
        pos = (12.0, 6.0)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        for _ in range(sup.k_hold + sup.confirm_frames + 2):
            sup.process_frame(self.noise(), np.zeros(2))
        self.assertFalse(sup.confirmed)
        self.assertEqual(sup.n_aborts, 1)
        self.assertIsNone(sup.hold_until)
        self.assertEqual(sup.n3_left, 0)
        self.assertFalse(sup.conf_active)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])

    def test_confirm_frames_independent_of_block_frames(self):
        """confirm_frames can be shorter than block_frames (presence's own block length)."""
        sup = make(ff_mode='hold', confirm=True, presence=True, z_thr=6.0, z_thr_local=6.0,
                   flux_thr=200.0, block_frames=5, confirm_frames=2, k_hold=2, n_cons=5, q_thr=1.1)
        pos = (10.0, 5.0)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertTrue(sup.conf_active)
        sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertIsNone(sup.confirmed)                    # 1 of 2 confirm frames
        sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertTrue(sup.confirmed)                      # confirmed well before a presence block completes


class TestWindowMaskCache(unittest.TestCase):

    def test_mask_is_cached_and_invalidated_on_window_change(self):
        sup = make()
        m1 = sup._window_mask()
        m1_again = sup._window_mask()
        self.assertIs(m1, m1_again)                         # same window: cache hit
        sup.w = np.array([5.0, 5.0])
        m2 = sup._window_mask()
        self.assertIsNot(m1, m2)                            # window moved: cache must be recomputed
        self.assertFalse(np.allclose(cpuArray(m1), cpuArray(m2)))
        m2_again = sup._window_mask()
        self.assertIs(m2, m2_again)


class TestIsPresent(unittest.TestCase):
    """Direct, noise-free check of the combined-presence boolean logic."""

    def test_any_single_evidence_is_enough(self):
        sup = make(presence=True, z_thr=10.0, z_thr_local=5.0, flux_thr=100.0)
        self.assertTrue(sup._is_present(20.0, 0.0, 0.0))    # global only
        self.assertTrue(sup._is_present(0.0, 6.0, 0.0))     # local only
        self.assertTrue(sup._is_present(0.0, 0.0, 150.0))   # flux only
        self.assertFalse(sup._is_present(0.0, 0.0, 0.0))    # none

    def test_disabled_evidences_are_ignored(self):
        """z_thr_local / flux_thr = None (not set): the corresponding evidence never counts,
        however large the value passed in (guards against a None-threshold comparison bug)."""
        sup = make(presence=True, z_thr=10.0)               # local and flux disabled
        self.assertFalse(sup._is_present(0.0, 999.0, 0.0))
        self.assertFalse(sup._is_present(0.0, 0.0, 999.0))
        self.assertTrue(sup._is_present(11.0, 0.0, 0.0))


class TestPresenceRegisterFalseWithMovingCommand(unittest.TestCase):

    def test_dropout_and_release_track_correctly_while_the_command_ramps(self):
        """presence_register=False (recommended for tracking): a stationary-on-detector spot with a
        ramping own command must still show a normal dropout -> release cycle. n_cons is set far out
        of reach so the ramp cannot accidentally trigger a spurious window move."""
        rng = np.random.default_rng(3)

        def noise():
            return NOISE * rng.standard_normal((N, N))

        sup = make(presence=True, presence_register=False, z_thr=6.0, k_absent=3, k_present=2,
                   block_frames=3, n_cons=1000, q_thr=1.1)
        k = 0

        def step(bright):
            nonlocal k
            u = np.array([0.3, -0.2]) * k
            k += 1
            frame = (spot(0.0, 0.0) if bright else 0.0) + noise()
            return sup.process_frame(frame, u)

        for _ in range(9):
            step(True)
        self.assertFalse(sup.dropout)
        for _ in range(10):
            step(False)
        self.assertTrue(sup.dropout)
        for _ in range(9):
            step(True)
        self.assertFalse(sup.dropout)
        self.assertEqual(sup.n_moves, 0)                    # never a spurious move throughout


class TestPrecisionAndDeviceParity(unittest.TestCase):

    def test_precision_parity_of_local_z_and_window_flux(self):
        rng = np.random.default_rng(6)
        frames = [spot(5.0, -3.0, amp=80.0) + NOISE * rng.standard_normal((N, N)) for _ in range(8)]
        out = {}
        for prec in (0, 1):
            sup = make(presence=True, z_thr=50.0, z_thr_local=10.0, flux_thr=100.0,
                       block_frames=4, precision=prec)
            for f in frames:
                sup.process_frame(f, np.zeros(2))
            out[prec] = sup.last_block
        np.testing.assert_allclose(out[1], out[0], rtol=1e-3)

    @cpu_and_gpu
    def test_cpu_gpu_parity_of_last_block(self, target_device_idx, xp):
        rng = np.random.default_rng(4)
        frames = [spot(0.0, 0.0, amp=60.0) + NOISE * rng.standard_normal((N, N)) for _ in range(12)]
        ref = make(-1, presence=True, z_thr=50.0, z_thr_local=50.0, flux_thr=200.0, block_frames=4)
        sup = make(target_device_idx, presence=True, z_thr=50.0, z_thr_local=50.0, flux_thr=200.0,
                   block_frames=4)
        for f in frames:
            ref.process_frame(f, np.zeros(2))
            sup.process_frame(f, np.zeros(2))
        np.testing.assert_allclose(sup.last_block, ref.last_block, rtol=1e-5)


class TestTelemetryPresenceOff(unittest.TestCase):

    def test_last_block_stays_nan_when_presence_is_off(self):
        sup = make()
        rng = np.random.default_rng(1)
        for _ in range(5):
            sup.process_frame(spot(0.0, 0.0) + NOISE * rng.standard_normal((N, N)), np.zeros(2))
        self.assertTrue(all(np.isnan(v) for v in sup.last_block))


if __name__ == '__main__':
    unittest.main()
