import specula
specula.init(0)  # Default target device

import unittest
import os
import glob

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.processing_objects.sh import SH
from specula.data_objects.laser_launch_telescope import LaserLaunchTelescope
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.subap_data import SubapData
from specula.lib.make_mask import make_mask
from specula.processing_objects.sh_slopec import ShSlopec
from test.specula_testlib import cpu_and_gpu

class TestShSlopec(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        test_dir = os.path.dirname(__file__)
        for fpath in glob.glob(os.path.join(test_dir, 'ConvolutionKernel*.fits')):
            if os.path.isfile(fpath):
                os.remove(fpath)

    def get_sh(self, target_device_idx, xp, with_laser_launch=False):
        # pupil is 1m
        pixel_pupil = 20
        pixel_pitch = 0.05
        # 2x2 subapertures
        subap_on_diameter = 2
        # lambda is 500 nm and lambda/D is 0.206 arcsec so 0.1 means 2 pixels per lambda/D
        wavelengthInNm = 500
        pxscale_arcsec = 0.1
        # big subaperture to avoid edge effects
        subap_npx = 12
        t_seconds = 1.0
        t = int(1e9)*t_seconds  # Convert 1 second to simulation time step

        # ------------------------------------------------------------------------------
        # Set up inputs for ShSlopec
        idxs = {}
        map = {}
        mask_subap = np.ones((subap_on_diameter*subap_npx, subap_on_diameter*subap_npx))

        count = 0
        for i in range(subap_on_diameter):
            for j in range(subap_on_diameter):
                mask_subap *= 0
                mask_subap[i*subap_npx:(i+1)*subap_npx,j*subap_npx:(j+1)*subap_npx] = 1
                idxs[count] = np.where(mask_subap == 1)
                map[count] = j * subap_on_diameter + i
                count += 1

        v = np.zeros((len(idxs), subap_npx*subap_npx), dtype=int)
        m = np.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = np.ravel_multi_index(idx, mask_subap.shape)
            m[k] = map[k]

        # "simple" SH
        if not with_laser_launch:
            sh = SH(wavelengthInNm=wavelengthInNm,
                    subap_wanted_fov=subap_npx * pxscale_arcsec,
                    sensor_pxscale=pxscale_arcsec,
                    subap_on_diameter=subap_on_diameter,
                    subap_npx=subap_npx,
                    target_device_idx=target_device_idx)

        # SH with laser launch
        else:
            laser_launch_tel = LaserLaunchTelescope(spot_size=pxscale_arcsec,
                                target_device_idx=target_device_idx)

            sh = SH(wavelengthInNm=wavelengthInNm,
                    subap_wanted_fov=subap_npx * pxscale_arcsec,
                    sensor_pxscale=pxscale_arcsec,
                    subap_on_diameter=subap_on_diameter,
                    subap_npx=subap_npx,
                    laser_launch_tel=laser_launch_tel,
                    target_device_idx=target_device_idx)

        flat_ef = ElectricField(pixel_pupil, pixel_pupil, pixel_pitch, S0=1, target_device_idx=target_device_idx)
        flat_ef.generation_time = t

        subapdata = SubapData(idxs=v, display_map = m, nx=subap_on_diameter, ny=subap_on_diameter, target_device_idx=target_device_idx)

        return sh, v, m, flat_ef, subapdata

    def get_single_subap_data(self, target_device_idx, np_sub):
        """
        Build a minimal SubapData with a single subaperture covering the
        whole np_sub x np_sub pixel frame, with row-major pixel ordering.
        This lets a plain (np_sub, np_sub) numpy array be compared directly
        against mask_weighted (also row-major) with no index bookkeeping.
        """
        mask_subap = np.ones((np_sub, np_sub))
        idx = np.where(mask_subap == 1)
        v = np.zeros((1, np_sub * np_sub), dtype=int)
        v[0] = np.ravel_multi_index(idx, mask_subap.shape)
        m = np.zeros(1, dtype=int)
        return SubapData(idxs=v, display_map=m, nx=1, ny=1, target_device_idx=target_device_idx)

    @cpu_and_gpu
    def test_windowed_flux_output_registered(self, target_device_idx, xp):
        """
        out_windowed_flux must be declared in output_names() and present
        (zero-initialized) in self.outputs right after construction.
        """
        np_sub = 8
        subapdata = self.get_single_subap_data(target_device_idx, np_sub)

        output_names = ShSlopec.output_names()
        self.assertIn('out_windowed_flux', output_names)
        self.assertIs(output_names['out_windowed_flux'].type, BaseValue)

        slopec = ShSlopec(subapdata, weightedPixRad=2.0, windowing=True,
                          target_device_idx=target_device_idx)

        self.assertIn('out_windowed_flux', slopec.outputs)
        self.assertIsInstance(slopec.outputs['out_windowed_flux'], BaseValue)
        np.testing.assert_array_equal(cpuArray(slopec.outputs['out_windowed_flux'].value),
                                      np.zeros(subapdata.n_subaps))

    @cpu_and_gpu
    def test_windowed_flux_value_and_generation_time(self, target_device_idx, xp):
        """
        out_windowed_flux must equal the WCoG-weighted flux (subap_tot),
        computed independently here with the same make_mask/make_xy
        formula used internally by computeXYweights(), and its
        generation_time must be updated after a trigger.
        """
        np_sub = 8
        weighted_pix_rad = 2.0
        subapdata = self.get_single_subap_data(target_device_idx, np_sub)

        # Independent expected mask: mirrors the "windowing" branch of
        # ShSlopec.computeXYweights() (hard-edged circular window).
        expected_mask = make_mask(np_sub, diaratio=(2.0 * weighted_pix_rad / np_sub), xp=np)

        pixel_values = np.arange(1, np_sub * np_sub + 1, dtype=float).reshape(np_sub, np_sub)
        expected_windowed_flux = np.sum(pixel_values * expected_mask)

        pixels = Pixels(np_sub, np_sub, target_device_idx=target_device_idx)
        pixels.pixels = xp.array(pixel_values)
        t = 1
        pixels.generation_time = t

        slopec = ShSlopec(subapdata, weightedPixRad=weighted_pix_rad, windowing=True,
                          target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        windowed_flux = slopec.outputs['out_windowed_flux'].value
        np.testing.assert_allclose(cpuArray(windowed_flux), expected_windowed_flux, rtol=1e-6)

        self.assertEqual(slopec.outputs['out_windowed_flux'].generation_time, t)

    @cpu_and_gpu
    def test_windowed_flux_differs_from_raw_flux(self, target_device_idx, xp):
        """
        out_windowed_flux (WCoG-weighted, local-SNR proxy) must differ from
        out_flux_per_subaperture (raw, unweighted sum over the whole
        subaperture) when weightedPixRad restricts the window to a small
        central region and there is signal/background outside that window
        (e.g. an acquisition-field corner far from the spot).
        """
        np_sub = 8
        weighted_pix_rad = 1.0
        subapdata = self.get_single_subap_data(target_device_idx, np_sub)

        pixel_values = np.zeros((np_sub, np_sub))
        # Signal inside the WCoG window (central 2x2 block, see below).
        pixel_values[3:5, 3:5] = 10.0
        # Background flux far from the window center: included in the raw
        # flux, excluded by the small weightedPixRad window.
        pixel_values[0, 0] = 1000.0

        # Sanity check: reproduce the window with the same formula used by
        # ShSlopec.computeXYweights() and confirm the corner pixel is
        # outside of it while the central block is inside.
        expected_mask = make_mask(np_sub, diaratio=(2.0 * weighted_pix_rad / np_sub), xp=np)
        self.assertEqual(expected_mask[0, 0], 0)
        self.assertTrue(np.all(expected_mask[3:5, 3:5] == 1))
        expected_windowed_flux = np.sum(pixel_values * expected_mask)
        expected_raw_flux = np.sum(pixel_values)
        self.assertNotEqual(expected_windowed_flux, expected_raw_flux)

        pixels = Pixels(np_sub, np_sub, target_device_idx=target_device_idx)
        pixels.pixels = xp.array(pixel_values)
        t = 1
        pixels.generation_time = t

        slopec = ShSlopec(subapdata, weightedPixRad=weighted_pix_rad, windowing=True,
                          target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        windowed_flux = cpuArray(slopec.outputs['out_windowed_flux'].value)
        raw_flux = cpuArray(slopec.outputs['out_flux_per_subaperture'].value)

        np.testing.assert_allclose(windowed_flux, expected_windowed_flux, rtol=1e-6)
        np.testing.assert_allclose(raw_flux, expected_raw_flux, rtol=1e-6)

        # The whole point of out_windowed_flux: it must differ from the
        # inherited raw-subaperture flux whenever the window doesn't cover
        # the full subaperture.
        self.assertFalse(np.allclose(windowed_flux, raw_flux))

    @cpu_and_gpu
    def test_pixelscale_and_slopes(self, target_device_idx, xp):
        """
        Test that verifies both pixel scale and slope computation for SH.
        A tilt that shifts the spot by 1 pixel should produce a slope of 1/(sh.subap_npx/2).
        """
        # Flat wavefront
        # pupil is 1m
        pixel_pupil = 20
        pixel_pitch = 0.05
        t = 1
        pxscale_arcsec = 0.1
        subap_npx = 12

        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        # tilt corresponding to pxscale_arcsec
        tilt_value = np.radians(pixel_pupil * pixel_pitch * 1/(60*60) * pxscale_arcsec)
        tilt = np.linspace(-tilt_value / 2, tilt_value / 2, pixel_pupil)
        
        # Tilted wavefront
        flat_ef.phaseInNm[:] = xp.array(np.broadcast_to(tilt, (pixel_pupil, pixel_pupil))) * 1e9
        flat_ef.generation_time = t+1

        sh.check_ready(t+1)
        sh.trigger()
        sh.post_trigger()
        tilted = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*tilted.shape, target_device_idx=target_device_idx)
        pixels.pixels = tilted
        pixels.generation_time = t+1

        # Create the slope computer object
        slopec = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t+1)
        slopec.trigger()
        slopec.post_trigger()
        slopes = slopec.outputs['out_slopes']

        # Expected value: 1/(subap_npx/2)
        expected_slope = 1.0 / (subap_npx / 2)

        # All X slopes (all slopes are valid) should be close to the expected value
        np.testing.assert_allclose(cpuArray(slopes.xslopes), expected_slope, rtol=1e-2, atol=1e-2)

    @cpu_and_gpu
    def test_weight_int_pixel_dt(self, target_device_idx, xp):
        """
        Test that verifies both slope computation and pixel accumulation
        with a specific weight_int_pixel_dt.
        """
        
        weight_int_pixel_dt = 3.0
        t_seconds = 1.0
        t = int(1e9)*t_seconds  # Convert 1 second to simulation time step

        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity =sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        # Create the slope computer object with the given parameters
        slopec = ShSlopec(subapdata, weight_int_pixel_dt=weight_int_pixel_dt, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Simulate 3 frames with known values
        for i in range(4):
            current_time = t*(i+1)
            if i == 2:
                # shift the pixels to simulate a change
                pixels.pixels = xp.roll(pixels.pixels, shift=1, axis=0)
            pixels.generation_time = current_time
            slopec.check_ready(current_time)
            slopec.trigger()
            slopec.post_trigger()

        # After two steps, the weight map should be the average of the two frames
        # normalized to the maximum intensity
        last_weights = slopec.int_pixels_weight

        last_weights_2d = xp.zeros_like(pixels.pixels)
        last_weights_2d_flat = last_weights_2d.flatten()
        last_weights_2d_flat[slopec.subap_idx.flatten()] = last_weights.T.flatten()
        last_weights_2d = last_weights_2d_flat.reshape(last_weights_2d.shape)

        # the expected weights are the average of frames
        expected_weights = 2 * intensity + xp.roll(intensity, shift=1, axis=0)
        expected_weights = expected_weights / expected_weights.max()

        np.testing.assert_allclose(cpuArray(last_weights_2d), cpuArray(expected_weights), atol=1e-3)

        # Then compares slopec.int_pixels.pixels and slopec.pixels.pixels:
        # they must be equal because the accumulation was resetted
        expected_int_pixels = pixels.pixels.astype(slopec.dtype)
        np.testing.assert_allclose(cpuArray(slopec.int_pixels.pixels), cpuArray(expected_int_pixels), atol=1e-3)

    @cpu_and_gpu
    def test_weight_int_pixel_dt_window(self, target_device_idx, xp):
        """
        Test that verifies both slope computation and pixel accumulation
        with a specific weight_int_pixel_dt and window_int_pixel.
        """
        weight_int_pixel_dt = 2.0
        window_int_threshold = 1.0

        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=True)
        t_seconds = 1.0
        t = int(1e9)*t_seconds  # Convert 1 second to simulation time step

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity/intensity.max()*10
        pixels.generation_time = t

        # Create the slope computer object with the given parameters
        slopec = ShSlopec(subapdata, weight_int_pixel_dt=weight_int_pixel_dt, window_int_threshold=window_int_threshold,
                          window_int_pixel=True, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Simulate 2 frames with known values
        for i in range(2):
            current_time = t*(i+1)
            pixels.generation_time = current_time
            slopec.check_ready(current_time)
            slopec.trigger()
            slopec.post_trigger()

        # After two steps, the weight map should be 4 square of 4x4 pixels
        # and value of 1.0 in the square and 0 outside
        last_weights = slopec.int_pixels_weight

        last_weights_2d = xp.zeros_like(pixels.pixels)
        last_weights_2d_flat = last_weights_2d.flatten()
        last_weights_2d_flat[slopec.subap_idx.flatten()] = last_weights.T.flatten()
        last_weights_2d = last_weights_2d_flat.reshape(last_weights_2d.shape)

        expected_weights = xp.zeros_like(last_weights_2d)
        expected_weights[4:8,   4:8] = 1.0
        expected_weights[16:20, 16:20] = 1.0
        expected_weights[4:8,   16:20] = 1.0
        expected_weights[16:20, 4:8] = 1.0

        np.testing.assert_equal(cpuArray(last_weights_2d), cpuArray(expected_weights), err_msg="Weight map does not match expected values.")

    @cpu_and_gpu
    def test_vec_wei_pix_rad_t_uses_last_valid_time(self, target_device_idx, xp):
        """
        Test that vecWeiPixRadT selects the last valid row based on time.
        """
        t_sh = int(1e9)
        t_slopec = int(2e9)

        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t_sh)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t_slopec

        vec_wei_pix_rad_t = xp.asarray([
            [0.5, 0.2],
            [1.5, 1.0],
            [3.0, 3.0],
        ])

        slopec = ShSlopec(subapdata, vecWeiPixRadT=vec_wei_pix_rad_t,
                          target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t_slopec)
        slopec.trigger()
        slopec.post_trigger()

        # At t=2.0 s, rows at 0.2 s and 1.0 s are valid, so selected radius must be 1.5.
        self.assertAlmostEqual(float(slopec.weighted_pix_rad), 1.5)

        expected = ShSlopec(subapdata, weightedPixRad=1.5, target_device_idx=target_device_idx)
        np.testing.assert_allclose(cpuArray(slopec.mask_weighted),
                                   cpuArray(expected.mask_weighted), atol=1e-6)


    @cpu_and_gpu
    def test_shslopec_slopesnull(self, target_device_idx, xp):
        '''
        Test that a SH Slopec correctly subtracts slope nulls (non-interleaved)
        '''
        # Flat wavefront
        # pupil is 1m
        t = 1
        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        # Create the slope computer object with the given parameters
        sn = Slopes(slopes=xp.arange(len(m)*2), interleave=False, target_device_idx=target_device_idx)

        slopec1 = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec2 = ShSlopec(subapdata, sn=sn, target_device_idx=target_device_idx)

        slopec1.inputs['in_pixels'].set(pixels)
        slopec2.inputs['in_pixels'].set(pixels)
        slopec1.check_ready(1)
        slopec2.check_ready(1)
        slopec1.trigger()
        slopec2.trigger()
        slopec1.post_trigger()
        slopec2.post_trigger()
        slopes1 = slopec1.outputs['out_slopes']
        slopes2 = slopec2.outputs['out_slopes']

        np.testing.assert_array_almost_equal(cpuArray(slopes2.slopes),
                                             cpuArray(slopes1.slopes - sn.slopes))


    @cpu_and_gpu
    def test_shslopec_interleaved_slopesnull(self, target_device_idx, xp):
        '''
        Test that a SH Slopec correctly subtracts slope nulls (interleaved)
        '''

        # Flat wavefront
        # pupil is 1m
        t=1
        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        # Create the slope computer object with the given parameters

        sn = Slopes(slopes=xp.arange(len(m)*2), interleave=True, target_device_idx=target_device_idx)

        slopec1 = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec2 = ShSlopec(subapdata, sn=sn, target_device_idx=target_device_idx)

        slopec1.inputs['in_pixels'].set(pixels)
        slopec2.inputs['in_pixels'].set(pixels)
        slopec1.check_ready(1)
        slopec2.check_ready(1)
        slopec1.trigger()
        slopec2.trigger()
        slopec1.post_trigger()
        slopec2.post_trigger()
        slopes1 = slopec1.outputs['out_slopes']
        slopes2 = slopec2.outputs['out_slopes']

        np.testing.assert_array_almost_equal(cpuArray(slopes2.xslopes),
                                             cpuArray(slopes1.xslopes - sn.xslopes))

        np.testing.assert_array_almost_equal(cpuArray(slopes2.yslopes),
                                             cpuArray(slopes1.yslopes - sn.yslopes))

    @cpu_and_gpu
    def test_flux_outputs(self, target_device_idx, xp):
        """
        Test that verifies flux_per_subaperture, total_counts, and subap_counts outputs.
        """
        t = 1
        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        slopec = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        # Get outputs
        flux_per_subap = slopec.outputs['out_flux_per_subaperture'].value
        total_counts = slopec.outputs['out_total_counts'].value
        subap_counts = slopec.outputs['out_subap_counts'].value

        # Verify flux_per_subaperture has correct shape
        self.assertEqual(flux_per_subap.shape[0], len(m))

        # Verify total_counts is sum of all flux
        expected_total = xp.sum(flux_per_subap)
        np.testing.assert_almost_equal(cpuArray(total_counts[0]),
                                       cpuArray(expected_total), decimal=5)

        # Verify subap_counts is mean of flux_per_subaperture
        expected_mean = xp.mean(flux_per_subap)
        np.testing.assert_almost_equal(cpuArray(subap_counts[0]),
                                       cpuArray(expected_mean), decimal=5)

        # Verify all values are positive
        self.assertTrue(xp.all(flux_per_subap >= 0))
        self.assertTrue(total_counts[0] >= 0)
        self.assertTrue(subap_counts[0] >= 0)

        # Verify generation times are set
        self.assertEqual(slopec.outputs['out_flux_per_subaperture'].generation_time, t)
        self.assertEqual(slopec.outputs['out_total_counts'].generation_time, t)
        self.assertEqual(slopec.outputs['out_subap_counts'].generation_time, t)

    @cpu_and_gpu
    def test_slopec_interleave(self, target_device_idx, xp):
        """
        Test that verifies the interleave option in Slopec.
        """
        t = 1
        sh, v, m, flat_ef, subapdata = self.get_sh(target_device_idx, xp, with_laser_launch=False)

        sh.inputs['in_ef'].set(flat_ef)
        sh.setup()
        sh.check_ready(t)
        sh.trigger()
        sh.post_trigger()

        intensity = sh.outputs['out_i'].i.copy()

        # Compute slopes using ShSlopec
        pixels = Pixels(*intensity.shape, target_device_idx=target_device_idx)
        pixels.pixels = intensity
        pixels.generation_time = t

        # Non-interleaved
        slopec_non = ShSlopec(subapdata, interleave=False, target_device_idx=target_device_idx)
        slopec_non.inputs['in_pixels'].set(pixels)
        slopec_non.check_ready(t)
        slopec_non.trigger()
        slopec_non.post_trigger()

        # Interleaved
        slopec_int = ShSlopec(subapdata, interleave=True, target_device_idx=target_device_idx)
        slopec_int.inputs['in_pixels'].set(pixels)
        slopec_int.check_ready(t)
        slopec_int.trigger()
        slopec_int.post_trigger()

        # Verify that both compute the same xslopes and yslopes values
        np.testing.assert_array_almost_equal(cpuArray(slopec_non.slopes.xslopes),
                                             cpuArray(slopec_int.slopes.xslopes))
        np.testing.assert_array_almost_equal(cpuArray(slopec_non.slopes.yslopes),
                                             cpuArray(slopec_int.slopes.yslopes))

        # Verify that the internal layout is different (interleaved vs non-interleaved)
        self.assertFalse(slopec_non.slopes.interleave)
        self.assertTrue(slopec_int.slopes.interleave)

        # Verify indices are different
        np.testing.assert_array_equal(cpuArray(slopec_non.slopes.indx()),
                                      cpuArray(xp.arange(0, len(m))))
        np.testing.assert_array_equal(cpuArray(slopec_int.slopes.indx()),
                                      cpuArray(xp.arange(0, len(m)*2, 2)))
