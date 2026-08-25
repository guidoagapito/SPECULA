import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.adaptive_window_sh_slopec import AdaptiveWindowShSlopec
from test.specula_testlib import cpu_and_gpu


class TestAdaptiveWindowShSlopec(unittest.TestCase):
    """Focused tests for the two-step adaptive-window SH slopec."""

    def get_test_setup(self, target_device_idx, xp, subap_npx=32, n_sub_side=1):
        """Build a synthetic sub-aperture geometry and full CCD shape."""
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

    def generate_spots(self, ccd_shape, subapdata, xp, fwhm=1.5, flux=200.0,
                       bg=0.0, shift_dx=0.0, shift_dy=0.0, noise_std=0.0):
        """Generate Gaussian SH spots on all sub-apertures with optional shift/noise."""
        np_sub = subapdata.np_sub
        n_subaps = subapdata.n_subaps
        ccd = np.full(ccd_shape, bg, dtype=np.float32)
        cntrd = (np_sub - 1) / 2.0

        x = np.arange(np_sub) - cntrd - shift_dx
        y = np.arange(np_sub) - cntrd - shift_dy
        xx, yy = np.meshgrid(x, y)

        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
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
        """Run one processing step at timestamp ``t``."""
        pixels.pixels = frame
        pixels.generation_time = t
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

    @cpu_and_gpu
    def test_fading_loss_holds_radius(self, target_device_idx, xp):
        """When flux fades out, radius must be held (no search expansion)."""
        t = int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx=32)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveWindowShSlopec(
            subapdata,
            base_pix_rad=1.2,
            max_pix_rad=6.0,
            growth_gain=1.0,
            deadzone_pix=0.0,
            beta_up=0.8,
            beta_down=0.5,
            lost_frames_req=2,
            fading_flux_thr=0.1,
            target_device_idx=target_device_idx,
        )
        slopec.inputs['in_pixels'].set(pixels)

        shifted = self.generate_spots(ccd_shape, subapdata, xp, flux=1000.0, shift_dx=3.0)
        for i in range(1, 5):
            self._run_frame(slopec, pixels, shifted, t * i)

        radius_before_fade = float(cpuArray(slopec.radius_curr)[0])

        empty = xp.zeros(ccd_shape, dtype=xp.float32)
        self._run_frame(slopec, pixels, empty, t * 5)
        self._run_frame(slopec, pixels, empty, t * 6)

        radius_after_fade = float(cpuArray(slopec.radius_curr)[0])
        state_code = int(cpuArray(slopec.state_code)[0])

        self.assertEqual(state_code, AdaptiveWindowShSlopec.STATE_LOST_FADING)
        self.assertAlmostEqual(radius_after_fade, radius_before_fade, delta=1e-3)

    @cpu_and_gpu
    def test_kinematic_lost_can_expand_radius(self, target_device_idx, xp):
        """Kinematic loss with expand policy must drive radius to ``max_pix_rad``."""
        t = int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx=32)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveWindowShSlopec(
            subapdata,
            base_pix_rad=1.0,
            max_pix_rad=5.0,
            growth_gain=0.8,
            deadzone_pix=0.0,
            beta_up=1.0,
            beta_down=0.1,
            lost_frames_req=2,
            corr_snr_thr=100.0,
            fading_flux_thr=0.0,
            lost_behavior_kinematic='expand',
            target_device_idx=target_device_idx,
        )
        slopec.inputs['in_pixels'].set(pixels)

        # Flat bright frame: high flux but forced bad correlation threshold,
        # so this is classified as kinematic loss after lost_frames_req.
        bright = self.generate_spots(ccd_shape, subapdata, xp, flux=2000.0, shift_dx=0.0)
        self._run_frame(slopec, pixels, bright, t)
        self._run_frame(slopec, pixels, bright, 2 * t)

        state_code = int(cpuArray(slopec.state_code)[0])
        radius_now = float(cpuArray(slopec.radius_curr)[0])

        self.assertEqual(state_code, AdaptiveWindowShSlopec.STATE_LOST_KINEMATIC)
        self.assertAlmostEqual(radius_now, 5.0, delta=1e-6)

    @cpu_and_gpu
    def test_kinematic_lost_hold_does_not_expand_radius(self, target_device_idx, xp):
        """Kinematic loss with hold policy must not force expansion to ``max_pix_rad``."""
        t = int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx=32)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = AdaptiveWindowShSlopec(
            subapdata,
            base_pix_rad=1.0,
            max_pix_rad=5.0,
            growth_gain=0.0,
            deadzone_pix=0.0,
            beta_up=1.0,
            beta_down=1.0,
            lost_frames_req=2,
            corr_snr_thr=100.0,
            fading_flux_thr=0.0,
            lost_behavior_kinematic='hold',
            target_device_idx=target_device_idx,
        )
        slopec.inputs['in_pixels'].set(pixels)

        bright = self.generate_spots(ccd_shape, subapdata, xp, flux=2000.0, shift_dx=0.0)
        self._run_frame(slopec, pixels, bright, t)
        radius_before_lost = float(cpuArray(slopec.radius_curr)[0])
        self._run_frame(slopec, pixels, bright, 2 * t)

        state_code = int(cpuArray(slopec.state_code)[0])
        radius_after_lost = float(cpuArray(slopec.radius_curr)[0])

        self.assertEqual(state_code, AdaptiveWindowShSlopec.STATE_LOST_KINEMATIC)
        self.assertAlmostEqual(radius_after_lost, radius_before_lost, delta=1e-6)
        self.assertLess(radius_after_lost, 5.0)

    @cpu_and_gpu
    def test_gain_compensation_keeps_slope_scale(self, target_device_idx, xp):
        """Gain compensation should keep slope scale close to geometric expectation."""
        t = int(1e9)
        subap_npx = 32
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx=subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        shift_x = 2.0
        shifted = self.generate_spots(ccd_shape, subapdata, xp, flux=2000.0, shift_dx=shift_x)
        expected = shift_x / (subap_npx / 2.0)

        slopec = AdaptiveWindowShSlopec(
            subapdata,
            base_pix_rad=0.8,
            max_pix_rad=5.0,
            growth_gain=1.2,
            deadzone_pix=0.0,
            beta_up=0.5,
            beta_down=0.1,
            gain_comp_enable=True,
            target_device_idx=target_device_idx,
        )
        slopec.inputs['in_pixels'].set(pixels)

        for i in range(1, 8):
            self._run_frame(slopec, pixels, shifted, i * t)

        slope_x = float(cpuArray(slopec.outputs['out_slopes'].xslopes)[0])
        self.assertAlmostEqual(slope_x, expected, delta=0.03)

    @cpu_and_gpu
    def test_adaptive_disabled_matches_sh_slopec(self, target_device_idx, xp):
        """With adaptive mode disabled, outputs must match baseline ``ShSlopec``."""
        t = int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx=24)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        frame = self.generate_spots(ccd_shape, subapdata, xp, flux=500.0, shift_dx=1.0)
        pixels.pixels = frame
        pixels.generation_time = t

        sh = ShSlopec(subapdata, weightedPixRad=0.0, target_device_idx=target_device_idx)
        sh.inputs['in_pixels'].set(pixels)
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        aw = AdaptiveWindowShSlopec(subapdata,
                                    adaptive_window_enable=False,
                                    base_pix_rad=0.0,
                                    max_pix_rad=0.0,
                                    weightedPixRad=0.0,
                                    target_device_idx=target_device_idx)
        aw.inputs['in_pixels'].set(pixels)
        aw.check_ready(t)
        aw.trigger()
        aw.post_trigger()

        np.testing.assert_allclose(cpuArray(aw.outputs['out_slopes'].xslopes),
                                   cpuArray(sh.outputs['out_slopes'].xslopes), atol=1e-6)
        np.testing.assert_allclose(cpuArray(aw.outputs['out_slopes'].yslopes),
                                   cpuArray(sh.outputs['out_slopes'].yslopes), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
