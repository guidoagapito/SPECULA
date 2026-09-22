import specula
specula.init(0)  # Default target device

import unittest
import warnings

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.pixels import Pixels
from specula.processing_objects.spot_supervisor import SpotSupervisor, NM_TO_PX
from test.specula_testlib import cpu_and_gpu

N = 64
RADIUS = 2.0
SIGMA_W = 2.0 * RADIUS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
Z_THR = 6.0


def spot(sx, sy, amp=300.0, sigma=1.5):
    """Gaussian spot at (sx, sy) px from the frame centre (x = column, y = row)."""
    c = (N - 1) / 2.0
    cols = np.exp(-0.5 * ((np.arange(N) - c - sx) / sigma) ** 2)
    rows = np.exp(-0.5 * ((np.arange(N) - c - sy) / sigma) ** 2)
    return amp * np.outer(rows, cols)


def make(target_device_idx=-1, **kw):
    return SpotSupervisor(weighted_pix_rad=RADIUS, np_sub=N, target_device_idx=target_device_idx, **kw)


def noisy(rng, frame=None, noise=3.0):
    base = 0.0 if frame is None else frame
    return base + noise * rng.standard_normal((N, N))


def feed(sup, rng, position, n_frames, u=(0.0, 0.0), noise=3.0):
    info = None
    for _ in range(n_frames):
        info = sup.process_frame(noisy(rng, spot(*position), noise), np.asarray(u, float))
    return info


def dark(sup, rng, n_frames):
    """Star off: noise only."""
    info = None
    for _ in range(n_frames):
        info = sup.process_frame(noisy(rng), np.zeros(2))
    return info


def physical(sup, rng, d, n_frames, u_loop=(0.0, 0.0), noise=1.0):
    """Closed-loop-like frames: the spot sits at d - u_loop - u_ff (disturbance minus our corrections).
    Returns the list of u_ff snapshots taken after every frame."""
    hist = []
    for _ in range(n_frames):
        pos = np.asarray(d, float) - np.asarray(u_loop, float) - sup.u_ff
        sup.process_frame(noisy(rng, spot(*pos), noise), np.asarray(u_loop, float))
        hist.append(sup.u_ff.copy())
    return hist


def n_ff_steps(hist, start=None):
    prev = np.zeros(2) if start is None else start
    count = 0
    for h in hist:
        if not np.allclose(h, prev):
            count += 1
        prev = h
    return count


def trigger_once(sup, frame, cmd, t, target_device_idx=-1, xp=np):
    pixels = Pixels(N, N, target_device_idx=target_device_idx)
    pixels.pixels = xp.asarray(frame)
    pixels.generation_time = t
    c = BaseValue(value=xp.asarray(cmd, dtype=float), target_device_idx=target_device_idx)
    c.generation_time = t
    sup.inputs['in_pixels'].set(pixels)
    sup.inputs['in_command'].set(c)
    sup.check_ready(t)
    sup.trigger()
    sup.post_trigger()


class TestSpotSupervisorExtra(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(11)

    # ---- construction / failure modes ----

    def test_defaults(self):
        sup = make()
        self.assertEqual(sup.k_hold, sup.n_cons)                       # default hold length
        self.assertAlmostEqual(sup.sigma_w, SIGMA_W)
        self.assertAlmostEqual(sup.delta, 1.5 * SIGMA_W)
        np.testing.assert_allclose(sup.cmd_to_px, np.eye(2) * NM_TO_PX)
        np.testing.assert_allclose(sup.px_to_cmd @ sup.cmd_to_px, np.eye(2), atol=1e-9)

    def test_explicit_k_hold_overrides_default(self):
        self.assertEqual(make(k_hold=2).k_hold, 2)
        self.assertEqual(make(k_hold=0).k_hold, 0)                     # 0 must not fall back to n_cons

    def test_singular_command_matrix_raises(self):
        with self.assertRaises(np.linalg.LinAlgError):
            make(cmd_to_px=[[1.0, 2.0], [2.0, 4.0]])

    def test_bad_command_matrix_shape_raises(self):
        with self.assertRaises(ValueError):
            make(cmd_to_px=[1.0, 2.0, 3.0])

    def test_frame_shape_mismatch_raises(self):
        sup = make()
        with self.assertRaises(ValueError):
            sup.process_frame(np.zeros((N // 2, N // 2)), np.zeros(2))

    def test_input_and_output_names(self):
        self.assertEqual(set(SpotSupervisor.input_names()), {'in_pixels', 'in_command'})
        self.assertEqual(set(SpotSupervisor.output_names()), {'out_window', 'out_feedforward', 'out_state'})

    # ---- process_frame contract ----

    def test_inputs_are_not_mutated(self):
        sup = make(ff_mode='one')
        frame = noisy(self.rng, spot(10.0, 5.0))
        frame_copy, u = frame.copy(), np.array([0.5, -0.5])
        u_copy = u.copy()
        sup.process_frame(frame, u)
        np.testing.assert_array_equal(frame, frame_copy)
        np.testing.assert_array_equal(u, u_copy)

    def test_integer_frames_are_accepted(self):
        sup = make()
        frame = np.round(spot(10.0, -6.0, amp=800.0)).astype(np.uint16)
        info = sup.process_frame(frame, np.zeros(2))
        np.testing.assert_allclose(info['est'], [10.0, -6.0], atol=1.0)

    def test_single_precision_matches_double(self):
        frames = [noisy(self.rng, spot(12.0, -9.0)) for _ in range(8)]
        out = {}
        for prec in (0, 1):
            sup = make(ff_mode='one', precision=prec)
            for f in frames:
                info = sup.process_frame(f, np.zeros(2))
            out[prec] = (info['est'], sup.u_ff.copy(), sup.n_moves)
        np.testing.assert_allclose(out[1][0], out[0][0], atol=0.5)
        np.testing.assert_allclose(out[1][1], out[0][1], atol=0.5)
        self.assertEqual(out[1][2], out[0][2])

    @cpu_and_gpu
    def test_cpu_gpu_parity_of_a_full_acquisition(self, target_device_idx, xp):
        rng = np.random.default_rng(5)
        frames = [noisy(rng, spot(-11.0, 8.0)) for _ in range(7)]
        ref = make(-1, ff_mode='hold')
        sup = make(target_device_idx, ff_mode='hold')
        for f in frames:
            ir, ig = ref.process_frame(f, np.zeros(2)), sup.process_frame(f, np.zeros(2))
            np.testing.assert_allclose(ig['est'], ir['est'])
            self.assertAlmostEqual(ig['ratio'], ir['ratio'], places=4)
        np.testing.assert_allclose(sup.w, ref.w)
        self.assertEqual(sup.n_moves, ref.n_moves)

    def test_dark_frame_is_safe(self):
        sup = make()
        info = sup.process_frame(np.zeros((N, N)), np.zeros(2))
        self.assertTrue(np.all(np.isfinite(info['est'])))
        self.assertEqual(info['ratio'], 1.0)                           # gl <= 0 -> guard vetoes
        for _ in range(sup.n_cons + 2):
            sup.process_frame(np.zeros((N, N)), np.zeros(2))
        self.assertEqual(sup.n_moves, 0)

    def test_reset_state_clears_dynamic_state(self):
        sup = make(ff_mode='hold')
        feed(sup, self.rng, (12.0, -9.0), sup.n_cons)
        self.assertNotEqual(np.abs(sup.w).sum(), 0.0)
        sup.reset_state()
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])
        self.assertEqual((sup.hist, sup.ratios, sup.hold_until, sup.frame), ([], [], None, 0))
        self.assertFalse(sup.dropout)
        # and it works again
        feed(sup, self.rng, (-8.0, 5.0), sup.n_cons)
        np.testing.assert_allclose(sup.w, [-8.0, 5.0], atol=1.0)

    # ---- consensus / thresholds ----

    def test_outlier_blocks_consensus_until_it_leaves_the_window(self):
        sup = make(ff_mode='one')
        pos = (12.0, 4.0)
        feed(sup, self.rng, pos, sup.n_cons - 1)
        feed(sup, self.rng, (-15.0, 10.0), 1)                          # outlier: no consensus on the last n_cons
        self.assertEqual(sup.n_moves, 0)
        feed(sup, self.rng, pos, sup.n_cons - 1)                       # outlier still inside the window
        self.assertEqual(sup.n_moves, 0)
        feed(sup, self.rng, pos, 1)                                    # n_cons clean frames since the outlier
        self.assertEqual(sup.n_moves, 1)

    def test_single_frame_consensus_with_n_cons_one(self):
        sup = make(ff_mode='one', n_cons=1)
        info = feed(sup, self.rng, (14.0, -3.0), 1)
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.u_ff, info['est'])

    def test_delta_threshold_with_guard_disabled(self):
        """With q_thr > 1 the guard never vetoes, so only ``far`` (> delta) decides."""
        near, far = 1.5, 4.0                                           # delta = 1.5 * sigma_w = 2.55 px
        self.assertLess(near, make().delta)
        self.assertGreater(far, make().delta)
        sup = make(q_thr=1.1)
        feed(sup, self.rng, (near, 0.0), 12)
        self.assertEqual(sup.n_moves, 0)
        sup = make(q_thr=1.1)
        feed(sup, self.rng, (far, 0.0), sup.n_cons)
        self.assertEqual(sup.n_moves, 1)

    def test_guard_radius_protects_a_spot_close_to_the_window(self):
        """Default guard: a spot beyond delta but inside r_loc (3 sigma_w = 5.1 px) is captured, no move."""
        sup = make()
        feed(sup, self.rng, (4.0, 0.0), 12)
        self.assertEqual(sup.n_moves, 0)
        sup = make()
        feed(sup, self.rng, (8.0, 0.0), sup.n_cons)
        self.assertEqual(sup.n_moves, 1)

    def test_guard_uses_the_median_of_the_ratios(self):
        """2 of 5 vetoing frames do not block the move, 3 of 5 do."""
        far = spot(15.0, 0.0, amp=300.0)
        near = spot(0.0, 0.0, amp=240.0)
        for n_near, expect in [(2, 1), (3, 0)]:
            sup = make(ff_mode='one')
            for k in range(sup.n_cons):
                frame = far + (near if k < n_near else 0.0) + 3.0 * self.rng.standard_normal((N, N))
                sup.process_frame(frame, np.zeros(2))
            self.assertEqual(sup.n_moves, expect, f'n_near={n_near}')

    def test_guard_ratio_without_pixels_near_the_window(self):
        sup = make()
        sup.w = np.array([500.0, 500.0])                               # window far outside the frame
        self.assertEqual(sup._guard_ratio(np.ones((N, N))), 1.0)

    def test_ref_offset_shifts_the_estimate_and_the_no_move_region(self):
        off = (3.0, -2.0)
        sup = make(ref_offset=list(off))
        info = feed(sup, self.rng, (12.0, -9.0), 1)
        np.testing.assert_allclose(info['est'], [9.0, -7.0], atol=1.0)
        sup = make(ref_offset=list(off))
        feed(sup, self.rng, off, 20)                                   # spot on the reference: nothing to do
        self.assertEqual(sup.n_moves, 0)
        sup = make(ref_offset=list(off), ff_mode='one')
        feed(sup, self.rng, (12.0, -9.0), sup.n_cons)
        np.testing.assert_allclose(sup.u_ff, [9.0, -7.0], atol=1.0)    # feedforward is relative to the reference

    def test_guard_window_is_centred_on_the_reference_offset(self):
        """A weaker spot sitting on the reference (offset from the frame centre) captures the window: no move."""
        off = (10.0, 0.0)
        far = spot(-15.0, 0.0, amp=300.0)
        on_ref = spot(*off, amp=240.0)
        for with_ref_spot, expect in [(True, 0), (False, 1)]:
            sup = make(ref_offset=list(off), ff_mode='one')
            for _ in range(sup.n_cons):
                frame = far + (on_ref if with_ref_spot else 0.0) + 3.0 * self.rng.standard_normal((N, N))
                sup.process_frame(frame, np.zeros(2))
            self.assertEqual(sup.n_moves, expect, f'with_ref_spot={with_ref_spot}')

    def test_held_window_is_the_centroid_displacement_whatever_ref_offset(self):
        """ref_offset is the peak-to-centroid offset for a spot at the reference: a frame whose peak sits at
        displacement + ref_offset must give a window at the displacement itself (no further correction)."""
        off = np.array([3.0, -2.0])
        pos = np.array([12.0, -9.0])
        sup = make(ref_offset=off.tolist(), ff_mode='hold')
        for t in range(1, sup.n_cons + 1):
            trigger_once(sup, noisy(self.rng, spot(*(pos + off))), [0.0, 0.0], t)
        win = cpuArray(sup.outputs['out_window'].value)
        np.testing.assert_allclose(win[:2], pos, atol=1.0)

    # ---- hold / n3 / multiple moves ----

    def test_zero_hold_applies_the_feedforward_immediately(self):
        sup = make(ff_mode='hold', k_hold=0)
        pos = (11.0, 6.0)
        feed(sup, self.rng, pos, sup.n_cons)
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.u_ff, pos, atol=1.0)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        self.assertIsNone(sup.hold_until)

    def test_hold_lasts_exactly_k_hold_frames(self):
        sup = make(ff_mode='hold', k_hold=4)
        pos = (10.0, -7.0)
        feed(sup, self.rng, pos, sup.n_cons)
        w_held = sup.w.copy()
        for k in range(sup.k_hold):
            np.testing.assert_array_equal(sup.w, w_held)
            np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0], err_msg=f'frame {k}')
            feed(sup, self.rng, pos, 1)
        np.testing.assert_allclose(sup.u_ff, w_held)

    def test_second_loss_during_the_hold_replaces_the_first(self):
        """New consensus while holding: window and hold timer restart, feedforward is not accumulated."""
        sup = make(ff_mode='hold', k_hold=20)
        p1, p2 = (12.0, 5.0), (-13.0, -8.0)
        feed(sup, self.rng, p1, sup.n_cons)
        self.assertEqual(sup.n_moves, 1)
        feed(sup, self.rng, p2, sup.n_cons)
        self.assertEqual(sup.n_moves, 2)
        np.testing.assert_allclose(sup.w, p2, atol=1.0)
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])
        feed(sup, self.rng, p2, sup.k_hold + 1)
        np.testing.assert_allclose(sup.u_ff, p2, atol=1.0)             # p2 only, not p1 + p2
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])

    def test_full_acquisition_is_stable_in_every_mode(self):
        """Physical loop: after the correction the spot is on the reference and no further move fires."""
        d = np.array([13.0, -9.0])
        for mode in ('hold', 'one', 'n3'):
            sup = make(ff_mode=mode)
            physical(sup, self.rng, d, 60)
            self.assertEqual(sup.n_moves, 1, mode)
            np.testing.assert_allclose(sup.u_ff, d, atol=1.0, err_msg=mode)
            np.testing.assert_array_equal(sup.w, [0.0, 0.0])

    def test_consecutive_moves_accumulate_the_feedforward(self):
        d1, d2 = np.array([12.0, 6.0]), np.array([-10.0, -9.0])
        for mode in ('hold', 'one', 'n3'):
            sup = make(ff_mode=mode)
            physical(sup, self.rng, d1, 40)
            physical(sup, self.rng, d2, 60)
            self.assertEqual(sup.n_moves, 2, mode)
            np.testing.assert_allclose(sup.u_ff, d2, atol=1.0, err_msg=mode)

    def test_n3_steps_are_two_frames_apart(self):
        sup = make(ff_mode='n3')
        d = np.array([12.0, 6.0])
        hist = physical(sup, self.rng, d, 30)
        steps = [k for k in range(1, len(hist)) if not np.allclose(hist[k], hist[k - 1])]
        self.assertEqual(len(steps), 3)
        self.assertEqual(np.diff(steps).tolist(), [2, 2])

    def test_n3_interrupted_by_a_new_consensus_restarts_cleanly(self):
        """Spot lost again after the first n3 step: old sequence is dropped, new one brings u_ff to d2."""
        sup = make(ff_mode='n3', n_cons=3, k_hold=0)
        d1, d2 = np.array([12.0, 6.0]), np.array([-12.0, -8.0])
        h1 = physical(sup, self.rng, d1, 3)                            # consensus at frame 3, first n3 step at once
        self.assertEqual(sup.n_moves, 1)
        self.assertEqual(sup.n3_left, 2)
        h2 = physical(sup, self.rng, d2, 30)
        self.assertEqual(sup.n_moves, 2)
        self.assertEqual(sup.n3_left, 0)
        self.assertIsNone(sup.hold_until)
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        np.testing.assert_allclose(sup.u_ff, d2, atol=1.0)
        # 2 steps of the interrupted sequence + 3 of the new one (the third old step never happens)
        self.assertEqual(n_ff_steps(h1 + h2), 5)

    # ---- presence / dropout ----

    def test_dropout_starts_on_the_kth_absent_frame_and_est_is_nan(self):
        sup = make(presence=True, z_thr=Z_THR, k_absent=3, k_present=2)
        feed(sup, self.rng, (0.0, 0.0), 5)
        flags = []
        for _ in range(3):
            info = dark(sup, self.rng, 1)
            flags.append(info['dropout'])
        self.assertEqual(flags, [False, False, True])
        self.assertTrue(np.all(np.isnan(info['est'])))
        self.assertTrue(np.isnan(info['ratio']))
        self.assertLess(info['z'], Z_THR)

    def test_isolated_present_block_resets_the_absent_counter(self):
        sup = make(presence=True, z_thr=Z_THR, k_absent=3, k_present=2)
        for _ in range(4):
            dark(sup, self.rng, 2)
            feed(sup, self.rng, (0.0, 0.0), 1)
        self.assertFalse(sup.dropout)

    def test_isolated_absent_block_resets_the_present_counter(self):
        sup = make(presence=True, z_thr=Z_THR, k_absent=2, k_present=3)
        dark(sup, self.rng, 4)
        self.assertTrue(sup.dropout)
        for _ in range(4):
            feed(sup, self.rng, (0.0, 0.0), 2)
            dark(sup, self.rng, 1)
        self.assertTrue(sup.dropout)

    def test_release_frame_has_estimate_and_hist_restarts(self):
        sup = make(presence=True, z_thr=Z_THR, k_absent=2, k_present=3, ff_mode='one')
        dark(sup, self.rng, 3)
        self.assertTrue(sup.dropout)
        pos = (12.0, -7.0)
        infos = [feed(sup, self.rng, pos, 1) for _ in range(3)]
        self.assertEqual([i['dropout'] for i in infos], [True, True, False])
        self.assertTrue(np.all(np.isfinite(infos[2]['est'])))
        self.assertEqual(len(sup.hist), 1)                             # history restarted on release
        feed(sup, self.rng, pos, sup.n_cons - 2)
        self.assertEqual(sup.n_moves, 0)
        feed(sup, self.rng, pos, 1)
        self.assertEqual(sup.n_moves, 1)

    def test_pre_dropout_estimates_do_not_leak_into_the_new_consensus(self):
        """Frames collected before a dropout must not combine with post-release frames."""
        sup = make(presence=True, z_thr=Z_THR, k_absent=2, k_present=1, ff_mode='one', n_cons=4)
        feed(sup, self.rng, (14.0, 8.0), 3)                            # 3 of 4 frames, no consensus yet
        self.assertEqual(sup.n_moves, 0)
        dark(sup, self.rng, 2)
        self.assertTrue(sup.dropout)
        feed(sup, self.rng, (14.0, 8.0), 1)                            # release, hist = 1 frame
        self.assertFalse(sup.dropout)
        self.assertEqual(sup.n_moves, 0)
        feed(sup, self.rng, (14.0, 8.0), 2)
        self.assertEqual(sup.n_moves, 0)
        feed(sup, self.rng, (14.0, 8.0), 1)
        self.assertEqual(sup.n_moves, 1)

    def test_dark_frames_count_as_absent(self):
        sup = make(presence=True, z_thr=Z_THR, k_absent=2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)            # std = 0 on an all-zero map
            for _ in range(2):
                info = sup.process_frame(np.zeros((N, N)), np.zeros(2))
        self.assertTrue(info['dropout'])

    def test_block_frames_evaluates_presence_once_per_block(self):
        sup = make(presence=True, z_thr=Z_THR, block_frames=3, k_absent=2)
        zs = [feed(sup, self.rng, (0.0, 0.0), 1)['z'] for _ in range(6)]
        self.assertEqual([z is None for z in zs], [True, True, False, True, True, False])
        self.assertEqual(sup.acc_count, 0)
        self.assertGreater(zs[2], Z_THR)

    def test_dropout_counts_blocks_not_frames(self):
        sup = make(presence=True, z_thr=Z_THR, block_frames=3, k_absent=2)
        feed(sup, self.rng, (0.0, 0.0), 3)
        dark(sup, self.rng, 5)                                         # 1 full absent block + 2 frames
        self.assertFalse(sup.dropout)
        dark(sup, self.rng, 1)                                         # 2nd absent block
        self.assertTrue(sup.dropout)

    def test_block_registration_follows_the_command_on_both_axes(self):
        """A spot moving with our own command is coherently integrated: z stays close to the static one."""
        bf = 4
        static = make(presence=True, z_thr=Z_THR, block_frames=bf)
        z_static = feed(static, self.rng, (0.0, 0.0), bf)['z']
        moving = make(presence=True, z_thr=Z_THR, block_frames=bf)
        ramp = np.array([2.0, -1.5])                                   # px/frame, different on each axis
        z = None
        for k in range(bf):
            u = ramp * k
            info = moving.process_frame(noisy(self.rng, spot(*(-u))), u)
            z = info['z'] if info['z'] is not None else z
        self.assertGreater(z, Z_THR)
        self.assertGreater(z, 0.6 * z_static)

    def test_hold_is_kept_through_a_dropout_and_applied_once_afterwards(self):
        sup = make(presence=True, z_thr=Z_THR, k_absent=2, k_present=2, ff_mode='hold', k_hold=3)
        pos = np.array([12.0, -9.0])
        feed(sup, self.rng, pos, sup.n_cons)
        self.assertEqual(sup.n_moves, 1)
        w_held = sup.w.copy()
        dark(sup, self.rng, 8)                                         # hold timer expires during the dropout
        self.assertTrue(sup.dropout)
        np.testing.assert_array_equal(sup.w, w_held)                   # window parked where it was
        np.testing.assert_array_equal(sup.u_ff, [0.0, 0.0])            # no feedforward while blind
        self.assertIsNotNone(sup.hold_until)
        feed(sup, self.rng, pos, 6)
        self.assertFalse(sup.dropout)
        np.testing.assert_allclose(sup.u_ff, w_held)                   # applied exactly once
        np.testing.assert_array_equal(sup.w, [0.0, 0.0])
        self.assertEqual(sup.n_moves, 1)

    def test_n3_is_paused_by_a_dropout_and_resumes(self):
        sup = make(presence=True, z_thr=Z_THR, k_absent=2, k_present=2, ff_mode='n3', k_hold=0, n_cons=3)
        d = np.array([12.0, 6.0])
        physical(sup, self.rng, d, 3)                                  # move + first n3 step
        after_first = sup.u_ff.copy()
        self.assertEqual(sup.n3_left, 2)
        dark(sup, self.rng, 6)
        self.assertTrue(sup.dropout)
        np.testing.assert_array_equal(sup.u_ff, after_first)           # frozen during the dropout
        self.assertEqual(sup.n3_left, 2)
        hist = physical(sup, self.rng, d, 20)
        self.assertEqual(sup.n3_left, 0)
        np.testing.assert_allclose(sup.u_ff, d, atol=1.0)
        self.assertEqual(n_ff_steps(hist, start=after_first), 2)       # the two remaining steps
        self.assertEqual(sup.n_moves, 1)

    # ---- trigger / outputs ----

    @cpu_and_gpu
    def test_trigger_dropout_outputs(self, target_device_idx, xp):
        sup = make(target_device_idx, presence=True, z_thr=Z_THR, k_absent=2, k_present=2)
        for t in range(1, 4):
            trigger_once(sup, noisy(self.rng, spot(0.0, 0.0)), [0.0, 0.0], t, target_device_idx, xp)
        state = cpuArray(sup.outputs['out_state'].value)
        self.assertEqual(state[4], 0.0)
        for t in range(4, 7):
            trigger_once(sup, noisy(self.rng), [0.0, 0.0], t, target_device_idx, xp)
        win = cpuArray(sup.outputs['out_window'].value)
        state = cpuArray(sup.outputs['out_state'].value)
        self.assertEqual(win[2], 1.0)                                  # hold flag for ShSlopecMovable
        self.assertEqual(state[4], 1.0)
        self.assertTrue(np.isnan(state[0]) and np.isnan(state[1]))     # no estimate while blind
        self.assertTrue(np.isfinite(state[2]))                         # z is still reported
        self.assertTrue(np.isnan(state[3]))
        for name in ('out_window', 'out_feedforward', 'out_state'):
            self.assertEqual(sup.outputs[name].generation_time, 6)

    def test_trigger_state_reports_estimate_and_move_counter(self):
        sup = make(ff_mode='one')
        pos = (-9.0, 7.0)
        for t in range(1, sup.n_cons + 1):
            trigger_once(sup, noisy(self.rng, spot(*pos)), [0.0, 0.0], t)
            state = cpuArray(sup.outputs['out_state'].value)
        np.testing.assert_allclose(state[:2], pos, atol=1.0)
        self.assertTrue(np.isnan(state[2]))                            # presence off -> z is NaN
        self.assertEqual(state[5], 1.0)
        self.assertEqual(state[4], 0.0)

    def test_trigger_window_output_holds_the_estimate(self):
        sup = make(ff_mode='hold')
        pos = (10.0, 8.0)
        for t in range(1, sup.n_cons + 1):
            trigger_once(sup, noisy(self.rng, spot(*pos)), [0.0, 0.0], t)
        win = cpuArray(sup.outputs['out_window'].value)
        np.testing.assert_allclose(win[:2], pos, atol=1.0)
        self.assertEqual(win[2], 0.0)
        np.testing.assert_array_equal(cpuArray(sup.outputs['out_feedforward'].value), [0.0, 0.0])

    def test_trigger_registers_the_command_through_a_rotated_matrix(self):
        """Command in nm -> u = M @ cmd (px) with swapped axes and a flipped sign; feedforward returned in nm."""
        m = np.array([[0.0, -0.01], [0.02, 0.0]])
        sup = make(ff_mode='one', cmd_to_px=m.tolist())
        d = np.array([11.0, -7.0])
        cmd = np.array([300.0, 100.0])
        for k in range(sup.n_cons):
            c = cmd * k
            u = m @ c
            trigger_once(sup, noisy(self.rng, spot(*(d - u)), 1.0), c, k + 1)
        self.assertEqual(sup.n_moves, 1)
        u_last = m @ (cmd * (sup.n_cons - 1))
        ff_nm = cpuArray(sup.outputs['out_feedforward'].value)
        np.testing.assert_allclose(m @ ff_nm, d - u_last, atol=1.0)    # feedforward brings the spot to the reference

    def test_trigger_ignores_extra_command_components(self):
        sup = make(ff_mode='one')
        for t in range(1, sup.n_cons + 1):
            trigger_once(sup, noisy(self.rng, spot(10.0, 4.0)), [0.0, 0.0, 123.0, -5.0], t)
        self.assertEqual(sup.n_moves, 1)

    def test_outputs_are_updated_in_place(self):
        sup = make(ff_mode='one')
        refs = [sup.outputs[k] for k in ('out_window', 'out_feedforward', 'out_state')]
        vals = [r.value for r in refs]
        for t in range(1, sup.n_cons + 1):
            trigger_once(sup, noisy(self.rng, spot(10.0, 4.0)), [0.0, 0.0], t)
        for r, v in zip(refs, vals):
            self.assertIs(r.value, v)                                  # links already connected keep the same buffers


if __name__ == '__main__':
    unittest.main()
