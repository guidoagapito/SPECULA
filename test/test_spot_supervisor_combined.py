import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec_movable import ShSlopecMovable
from specula.processing_objects.spot_supervisor import SpotSupervisor
from test.specula_testlib import cpu_and_gpu

N = 64
RADIUS = 2.0
NOISE = 3.0


def spot(sx, sy, amp=300.0, sigma=1.5):
    c = (N - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(N) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(N) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def make(target_device_idx=-1, **kw):
    return SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=target_device_idx, **kw)


class TestCombinedPresence(unittest.TestCase):
    """Dropout only if global z, local z and window flux are all absent."""

    def setUp(self):
        self.rng = np.random.default_rng(11)
        # thresholds chosen for this toy frame: a faint spot in the window is below the global-z
        # threshold but clearly above the flux threshold
        self.z_thr, self.z_loc, self.f_thr = 50.0, 50.0, 200.0

    def noise(self):
        return NOISE * self.rng.standard_normal((N, N))

    def run_frames(self, sup, frames):
        for f in frames:
            sup.process_frame(f, np.zeros(2))

    def test_faint_star_in_window_is_not_a_dropout_with_flux(self):
        faint = [spot(0.0, 0.0, amp=60.0) + self.noise() for _ in range(20)]
        only_g = make(presence=True, z_thr=self.z_thr, k_absent=3)
        comb = make(presence=True, z_thr=self.z_thr, flux_thr=self.f_thr, k_absent=3)
        self.run_frames(only_g, faint)
        self.run_frames(comb, faint)
        self.assertTrue(only_g.dropout)                   # the global statistic alone calls it absent
        self.assertFalse(comb.dropout)                    # the window flux keeps it present
        z, z_loc, f = comb.last_block
        self.assertLess(z, self.z_thr)
        self.assertGreater(f, self.f_thr)

    def test_true_absence_is_still_detected(self):
        sup = make(presence=True, z_thr=6.0, z_thr_local=4.0, flux_thr=self.f_thr, k_absent=3)
        self.run_frames(sup, [spot(0.0, 0.0) + self.noise() for _ in range(5)])
        self.assertFalse(sup.dropout)
        self.run_frames(sup, [self.noise() for _ in range(4)])
        self.assertTrue(sup.dropout)

    def test_release_on_any_single_evidence(self):
        sup = make(presence=True, z_thr=self.z_thr, flux_thr=self.f_thr, k_absent=2, k_present=3)
        self.run_frames(sup, [self.noise() for _ in range(3)])
        self.assertTrue(sup.dropout)
        self.run_frames(sup, [spot(0.0, 0.0, amp=60.0) + self.noise() for _ in range(3)])
        self.assertFalse(sup.dropout)                    # flux alone releases (global z stays below its threshold)

    def test_lost_spot_elsewhere_is_not_a_dropout(self):
        """Spot far from the window: flux and local z absent, the global z keeps it present."""
        sup = make(presence=True, z_thr=6.0, z_thr_local=4.0, flux_thr=self.f_thr, k_absent=3, q_thr=-1.0)
        self.run_frames(sup, [spot(20.0, -15.0) + self.noise() for _ in range(10)])
        self.assertFalse(sup.dropout)
        z, z_loc, f = sup.last_block
        self.assertGreater(z, 6.0)
        self.assertLess(f, self.f_thr)

    def test_local_z_follows_the_window(self):
        sup = make(presence=True, z_thr=1e9, z_thr_local=4.0, k_absent=2)
        sup.w = np.array([10.0, 5.0])
        self.run_frames(sup, [spot(10.0, 5.0, amp=120.0) + self.noise() for _ in range(4)])
        self.assertFalse(sup.dropout)                    # only the local statistic can see it (global disabled)
        sup2 = make(presence=True, z_thr=1e9, z_thr_local=4.0, k_absent=2)
        self.run_frames(sup2, [spot(10.0, 5.0, amp=120.0) + self.noise() for _ in range(4)])
        self.assertTrue(sup2.dropout)                    # window at the reference: the spot is outside the disk

    def test_unregistered_blocks_do_not_smear_a_tracked_spot(self):
        """Tracking: spot stationary on the detector while the command follows the disturbance."""
        zs = {}
        for reg in (True, False):
            sup = make(presence=True, z_thr=0.0, block_frames=8, presence_register=reg)
            rng = np.random.default_rng(5)
            for k in range(8):
                sup.process_frame(spot(0.0, 0.0, amp=40.0) + NOISE * rng.standard_normal((N, N)),
                                  np.array([1.5 * k, -1.0 * k]))
            zs[reg] = sup.last_block[0]
        self.assertGreater(zs[False], zs[True] + 1.0)

    @cpu_and_gpu
    def test_window_flux_matches_sh_slopec_movable(self, target_device_idx, xp):
        """The supervisor's window flux is ShSlopecMovable's windowed flux (thr_value = 0)."""
        w = (6.0, -4.0)
        frame = spot(6.3, -3.8) + NOISE * np.random.default_rng(2).standard_normal((N, N))
        sup = make(target_device_idx, presence=True, z_thr=0.0, flux_thr=0.0)
        sup.w = np.array(w)
        sup.process_frame(frame, np.zeros(2))
        idx = np.where(np.ones((N, N)) == 1)
        v = np.zeros((1, N * N), dtype=int)
        v[0] = np.ravel_multi_index(idx, (N, N))
        subap = SubapData(idxs=v, display_map=np.zeros(1, dtype=int), nx=1, ny=1, target_device_idx=target_device_idx)
        slopec = ShSlopecMovable(subap, weightedPixRad=RADIUS, thr_value=0, target_device_idx=target_device_idx)
        pix = Pixels(N, N, target_device_idx=target_device_idx)
        pix.pixels = xp.asarray(frame)
        pix.generation_time = 1
        win = BaseValue(value=xp.asarray([w[0], w[1], 0.0]), target_device_idx=target_device_idx)
        win.generation_time = 1
        slopec.inputs['in_pixels'].set(pix)
        slopec.inputs['in_window'].set(win)
        slopec.check_ready(1)
        slopec.trigger()
        slopec.post_trigger()
        ref = float(cpuArray(slopec.outputs['out_windowed_flux'].value)[0])
        self.assertAlmostEqual(sup.last_block[2], ref, delta=1e-3 * abs(ref))

    def test_telemetry_columns(self):
        sup = make(presence=True, z_thr=6.0, z_thr_local=4.0, flux_thr=self.f_thr)
        pixels = Pixels(N, N, target_device_idx=-1)
        pixels.pixels = spot(0.0, 0.0) + self.noise()
        pixels.generation_time = 1
        cmd = BaseValue(value=np.zeros(2), target_device_idx=-1)
        cmd.generation_time = 1
        sup.inputs['in_pixels'].set(pixels)
        sup.inputs['in_command'].set(cmd)
        sup.check_ready(1)
        sup.trigger()
        sup.post_trigger()
        st = cpuArray(sup.outputs['out_state'].value)
        self.assertEqual(st.shape, (9,))
        np.testing.assert_allclose(st[[2, 6, 7]], sup.last_block, rtol=1e-5)
        self.assertEqual(st[8], 0.0)


class TestConfirmedHold(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(7)

    def noise(self):
        return NOISE * self.rng.standard_normal((N, N))

    def make_conf(self, **kw):
        base = dict(ff_mode='hold', confirm=True, z_thr_local=6.0, flux_thr=200.0, block_frames=4)
        base.update(kw)
        return make(**base)

    def test_confirmed_move_applies_the_feedforward(self):
        sup = self.make_conf()
        pos = (12.0, -9.0)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        w_held = sup.w.copy()
        for _ in range(max(sup.k_hold, sup.confirm_frames) + 2):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        self.assertTrue(sup.confirmed)
        np.testing.assert_allclose(sup.u_ff, w_held)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        self.assertEqual(sup.n_aborts, 0)

    def test_unconfirmed_move_is_aborted_and_the_window_restored(self):
        """A transient source triggers a consensus and vanishes: nothing at the new window -> abort."""
        sup = self.make_conf()
        for _ in range(sup.n_cons):
            sup.process_frame(spot(12.0, -9.0) + self.noise(), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        for _ in range(max(sup.k_hold, sup.confirm_frames) + 2):
            sup.process_frame(self.noise(), np.zeros(2))
        self.assertFalse(sup.confirmed)
        self.assertEqual(sup.n_aborts, 1)
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])       # no feedforward applied
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])          # back to the previous window
        self.assertIsNone(sup.hold_until)

    def test_feedforward_waits_for_the_confirmation_block(self):
        sup = self.make_conf(k_hold=1, confirm_frames=6)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(10.0, 4.0) + self.noise(), np.zeros(2))
        for k in range(5):                                         # hold expired, confirmation incomplete
            sup.process_frame(spot(10.0, 4.0) + self.noise(), np.zeros(2))
            np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])
        sup.process_frame(spot(10.0, 4.0) + self.noise(), np.zeros(2))
        sup.process_frame(spot(10.0, 4.0) + self.noise(), np.zeros(2))
        self.assertTrue(np.any(sup.u_ff != 0.0))

    def test_confirm_needs_local_thresholds(self):
        with self.assertRaises(ValueError):
            make(ff_mode='hold', confirm=True)

    def test_confirm_rejects_one_shot_mode(self):
        with self.assertRaises(ValueError):
            make(ff_mode='one', confirm=True, flux_thr=1.0)

    def test_n3_with_confirmation_starts_after_confirming(self):
        sup = self.make_conf(ff_mode='n3')
        pos = (9.0, 6.0)
        for _ in range(sup.n_cons):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        w0 = sup.w.copy()
        for _ in range(max(sup.k_hold, sup.confirm_frames) + 8):
            sup.process_frame(spot(*pos) + self.noise(), np.zeros(2))
        np.testing.assert_allclose(sup.u_ff, w0)
        self.assertEqual(sup.n_aborts, 0)


if __name__ == '__main__':
    unittest.main()
