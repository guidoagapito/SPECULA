import specula
specula.init(0)  # Default target device

import unittest

from specula import np, cpuArray

from specula.processing_objects.spot_supervisor import SpotSupervisor
from test.specula_testlib import cpu_and_gpu

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


class TestSpotSupervisorLookOutput(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(11)

    @cpu_and_gpu
    def test_look_image_is_the_mean_of_its_frames(self, target_device_idx, xp):
        """No command: the registration is the identity, so the look image is the plain mean of its frames."""
        sup = make(target_device_idx, look_frames=3)
        frames = [spot(7.0, -3.0) + 2.0 * self.rng.standard_normal((N, N)) for _ in range(3)]
        for f in frames:
            sup.process_frame(f, np.zeros(2))
        np.testing.assert_allclose(cpuArray(sup.last_look), np.mean(frames, axis=0), atol=1e-3)

    @cpu_and_gpu
    def test_look_image_is_registered_with_the_command(self, target_device_idx, xp):
        """Spot fixed in disturbance coordinates while the own command shifts it by 2 px per frame: the look image
        is in the detector frame of the first frame, so its peak is where the spot was in that frame."""
        sup = make(target_device_idx, look_frames=4)
        d, step = np.array([8.0, 4.0]), np.array([2.0, 0.0])
        for k in range(4):
            u = k * step                                   # command passed at step k ...
            applied = (k - 1) * step if k >= 1 else np.zeros(2)   # ... reaches the pixels at step k + 1 (L = 1)
            sup.process_frame(spot(*(d - applied)), u)
        img = cpuArray(sup.last_look)
        iy, ix = np.unravel_index(np.argmax(img), img.shape)
        c = (N - 1) / 2.0
        self.assertAlmostEqual(ix - c, d[0], delta=0.6)
        self.assertAlmostEqual(iy - c, d[1], delta=0.6)

    @cpu_and_gpu
    def test_disabled_supervisor_forms_looks_but_takes_no_decision(self, target_device_idx, xp):
        """A spot far from the window: disabled, no move and no feedforward (looks still find it); enabled, the
        consensus moves the window onto it."""
        sup = make(target_device_idx, look_frames=2, n_cons=2)
        sup.set_enabled(False)
        for _ in range(20):
            info = sup.process_frame(spot(-12.0, 9.0) + 2.0 * self.rng.standard_normal((N, N)), np.zeros(2))
        self.assertEqual(sup.n_moves, 0)
        np.testing.assert_allclose(sup.u_ff, [0.0, 0.0])
        np.testing.assert_allclose(info['est'], [-12.0, 9.0], atol=1.0)
        sup.set_enabled(True)
        for _ in range(4):
            sup.process_frame(spot(-12.0, 9.0) + 2.0 * self.rng.standard_normal((N, N)), np.zeros(2))
        self.assertEqual(sup.n_moves, 1)
        np.testing.assert_allclose(sup.w, [-12.0, 9.0], atol=1.0)
        for _ in range(sup.k_hold + 1):                               # hold, then the feedforward
            sup.process_frame(spot(-12.0, 9.0) + 2.0 * self.rng.standard_normal((N, N)), np.zeros(2))
        np.testing.assert_allclose(sup.u_ff, [-12.0, 9.0], atol=1.0)

    def test_hold_flag_while_disabled_only_if_requested(self):
        sup = make()
        sup.set_enabled(False)
        self.assertEqual(sup.window_value()[2], 0.0)
        sup = make(hold_when_disabled=True)
        sup.set_enabled(False)
        self.assertEqual(sup.window_value()[2], 1.0)          # loop open while disabled
        sup.set_enabled(True)
        self.assertEqual(sup.window_value()[2], 0.0)

    def test_a_step_without_new_pixels_is_not_processed(self):
        """Detector integrating two simulation steps: the supervisor must process each frame once, also when
        another input (here in_enable) is refreshed at every step."""
        from types import SimpleNamespace
        sup = make(look_frames=2)
        pixels = SimpleNamespace(pixels=spot(5.0, 0.0), generation_time=0)
        sup.local_inputs = {'in_pixels': pixels, 'in_command': SimpleNamespace(value=np.zeros(2), generation_time=0),
                            'in_enable': SimpleNamespace(value=np.ones(1), generation_time=0)}
        for step in range(6):
            sup.current_time = step
            pixels.generation_time = step - step % 2           # a new frame every two steps
            sup.trigger_code()
        self.assertEqual(sup.frame, 3)


if __name__ == '__main__':
    unittest.main()
