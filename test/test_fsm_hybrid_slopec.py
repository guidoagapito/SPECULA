import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
from specula.processing_objects.sh_slopec import ShSlopec
from specula.processing_objects.fsm_hybrid_slopec import FsmHybridSlopec
from test.specula_testlib import cpu_and_gpu


class TestFsmHybridSlopec(unittest.TestCase):

    def get_test_setup(self, target_device_idx, xp, subap_npx=32, n_sub_side=2):
        """
        Creates a dummy Shack-Hartmann sensor and associated data 
        to test the vectorized algorithm.
        """
        # Create a dummy index array for 4 sub-apertures (2x2)
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

        # Provide the total empty CCD array shape
        ccd_shape = (n_sub_side * subap_npx, n_sub_side * subap_npx)
        return subapdata, ccd_shape

    def generate_spots(self, ccd_shape, subapdata, xp, fwhm=1.5, flux=100.0,
                       bg=1.0, shift_dx=0.0, shift_dy=0.0, noise_std=0.0):
        """Generates Gaussian spots with an optional sub-pixel shift along the X axis."""
        np_sub = subapdata.np_sub
        n_subaps = subapdata.n_subaps
        ccd = np.full(ccd_shape, bg, dtype=np.float32)
        cntrd = (np_sub - 1) / 2.0

        x = np.arange(np_sub) - cntrd - shift_dx
        y = np.arange(np_sub) - cntrd - shift_dy
        xx, yy = np.meshgrid(x, y)

        sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gaussian = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        gaussian = (gaussian / np.sum(gaussian)) * flux

        # Insert the spot into each sub-aperture
        for k in range(n_subaps):
            # Retrieve 2D coordinates from the 1D index
            idx_1d = cpuArray(subapdata.idxs[k])
            iy, ix = np.unravel_index(idx_1d, ccd_shape)

            # Reconstruct the patch and add the gaussian
            min_y, max_y = np.min(iy), np.max(iy) + 1
            min_x, max_x = np.min(ix), np.max(ix) + 1
            ccd[min_y:max_y, min_x:max_x] += gaussian

        if noise_std > 0:
            ccd += np.random.normal(0, noise_std, ccd.shape).astype(np.float32)

        return xp.asarray(ccd)

    @cpu_and_gpu
    def test_fsm_subpixel_accuracy_and_y_axis(self, target_device_idx, xp):
        """Test accuracy with sub-pixel shifts on both axes to cover gain/offset bugs."""
        lock_req, subap_npx, t = 3, 32, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        shift_x, shift_y = 0.35, -0.65
        shifted_frame = self.generate_spots(ccd_shape, subapdata, xp,
                                            shift_dx=shift_x, shift_dy=shift_y)

        slopec_fsm = FsmHybridSlopec(subapdata, lock_frames_req=lock_req,
                                     target_device_idx=target_device_idx)
        slopec_fsm.inputs['in_pixels'].set(pixels)

        for i in range(1, lock_req + 2):
            pixels.pixels = shifted_frame
            pixels.generation_time = t * i
            slopec_fsm.check_ready(t * i)
            slopec_fsm.trigger()
            slopec_fsm.post_trigger()

        slopes_x = cpuArray(slopec_fsm.outputs['out_slopes'].xslopes)
        slopes_y = cpuArray(slopec_fsm.outputs['out_slopes'].yslopes)

        expected_slope_x = shift_x / (subap_npx / 2.0)
        expected_slope_y = shift_y / (subap_npx / 2.0)

        np.testing.assert_allclose(slopes_x, expected_slope_x, atol=1e-2,
                                   err_msg="X sub-pixel accuracy failed")
        np.testing.assert_allclose(slopes_y, expected_slope_y, atol=1e-2,
                                   err_msg="Y sub-pixel accuracy/sign failed")

    @cpu_and_gpu
    def test_fsm_off_center_hold(self, target_device_idx, xp):
        """Verifies zero-order hold correctly propagates off-center slopes during signal drop."""
        lock_req, t = 3, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, 32)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        shift_x = 2.0
        good_frame = self.generate_spots(ccd_shape, subapdata, xp, shift_dx=shift_x)
        bad_frame = xp.full(ccd_shape, 1.0, dtype=xp.float32)

        slopec = FsmHybridSlopec(subapdata, lock_frames_req=lock_req,
                                 target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Lock off-center
        for i in range(1, lock_req + 1):
            pixels.pixels = good_frame
            pixels.generation_time = t * i
            slopec.check_ready(t * i)
            slopec.trigger()
            slopec.post_trigger()

        # Drop signal
        pixels.pixels = bad_frame
        pixels.generation_time = t * (lock_req + 1)
        slopec.check_ready(pixels.generation_time)
        slopec.trigger()
        slopec.post_trigger()

        self.assertTrue(np.all(cpuArray(slopec.is_locked)),
                        "Lost lock immediately instead of holding.")

        slopes_x = cpuArray(slopec.outputs['out_slopes'].xslopes)
        expected_slope = shift_x / (32 / 2.0)
        np.testing.assert_allclose(slopes_x, expected_slope, atol=1e-3,
                                   err_msg="Zero-order hold failed to maintain off-center position")

    @cpu_and_gpu
    def test_fsm_ghost_flush_on_loss_of_lock(self, target_device_idx, xp):
        """Verifies EMA buffers are completely flushed upon loss of lock (preventing ghosting)."""
        lock_req, max_miss, t = 3, 5, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, 32)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        frame_pos1 = self.generate_spots(ccd_shape, subapdata, xp, shift_dx=-2.0)
        # Background fisso a 1.0
        frame_empty = xp.full(ccd_shape, 1.0, dtype=xp.float32)

        slopec = FsmHybridSlopec(subapdata, lock_frames_req=lock_req, max_missed_frames=max_miss,
                                 target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Lock at pos1
        for i in range(1, lock_req + 1):
            pixels.pixels = frame_pos1
            pixels.generation_time = t * i
            slopec.check_ready(t * i)
            slopec.trigger()
            slopec.post_trigger()

        # Fade out until hard drop
        for i in range(1, max_miss + 1):
            pixels.pixels = frame_empty
            pixels.generation_time = t * (lock_req + i)
            slopec.check_ready(pixels.generation_time)
            slopec.trigger()
            slopec.post_trigger()

        self.assertFalse(np.all(cpuArray(slopec.is_locked)), "FSM failed to drop lock")

        # Verify EMA pixel buffer is flattened to the background value (1.0), no ghosts from pos1
        ema_pix = cpuArray(slopec.ema_pixels)
        np.testing.assert_allclose(ema_pix, 1.0, atol=1e-3,
                                   err_msg="EMA buffer ghosting detected! Flush failed.")

    @cpu_and_gpu
    def test_fsm_low_snr_flickering(self, target_device_idx, xp):
        """Verifies acquisition and tracking robustness under heavy noise (low SNR regime)."""
        lock_req, t = 5, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, 32)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        slopec = FsmHybridSlopec(subapdata, lock_frames_req=lock_req, snr_thr=3.0,
                                 target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        for i in range(1, 20):
            # SNR fisicamente realistico per testare il Flickering:
            # Singolo frame borderline (potrebbe fallire), ma EMA accumula e supera la soglia.
            noisy_frame = self.generate_spots(ccd_shape, subapdata, xp,
                                              flux=60.0, bg=10.0, noise_std=8.0)
            pixels.pixels = noisy_frame
            pixels.generation_time = t * i
            slopec.check_ready(t * i)
            slopec.trigger()
            slopec.post_trigger()

        self.assertTrue(np.all(cpuArray(slopec.is_locked)),
                        "FSM failed to lock in low SNR regime via EMA integration")

    @cpu_and_gpu
    def test_fsm_kinematic_tracking_windshake(self, target_device_idx, xp):
        """Verifies the kinematic predictor can follow a moving spot (windshake simulation)."""
        lock_req, subap_npx, t = 3, 32, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        
        slopec = FsmHybridSlopec(subapdata, lock_frames_req=lock_req,prior_sigma=5.0,
                                 target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # 1. Lock on center
        good_frame = self.generate_spots(ccd_shape, subapdata, xp)
        for i in range(1, lock_req + 1):
            pixels.pixels = good_frame
            pixels.generation_time = t * i
            slopec.check_ready(t * i)
            slopec.trigger()
            slopec.post_trigger()

        # 2. Simulate Windshake (spot moves by 0.5 pixels every frame)
        for i in range(1, 4):
            moving_frame = self.generate_spots(ccd_shape, subapdata, xp, shift_dx=0.5 * i)
            pixels.pixels = moving_frame
            pixels.generation_time = t * (lock_req + i)
            slopec.check_ready(pixels.generation_time)
            slopec.trigger()
            slopec.post_trigger()
            
            slopes_x = cpuArray(slopec.outputs['out_slopes'].xslopes)
            expected_slope_x = (0.5 * i) / (subap_npx / 2.0)
            np.testing.assert_allclose(slopes_x, expected_slope_x, atol=1e-2, 
                                       err_msg=f"Kinematic tracker failed to follow moving spot at step {i}")

    @cpu_and_gpu
    def test_fsm_vs_sh_slopec_normalization(self, target_device_idx, xp):
        """Verifies slope scale normalization against standard ShSlopec under tracking conditions."""
        subap_npx = 32
        lock_frames_req = 3
        t = int(1e9)

        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp,
                                                   subap_npx=subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)

        # Generate a continuous sequence with a 1-pixel shift to the right (+X)
        shifted_frame = self.generate_spots(ccd_shape, subapdata, xp, fwhm=1.5,
                                            flux=1000.0, bg=0.0, shift_dx=1.0)

        # 1. Execute Standard ShSlopec
        pixels.pixels = shifted_frame
        pixels.generation_time = t
        slopec_sh = ShSlopec(subapdata, target_device_idx=target_device_idx)
        slopec_sh.inputs['in_pixels'].set(pixels)
        slopec_sh.check_ready(t)
        slopec_sh.trigger()
        slopec_sh.post_trigger()

        slopes_sh_x = cpuArray(slopec_sh.outputs['out_slopes'].xslopes)

        # 2. Execute FSM Hybrid Slopec
        # We MUST complete the bootstrap loop to engage tracking, otherwise output slopes are zeroed out.
        slopec_fsm = FsmHybridSlopec(subapdata, lock_frames_req=lock_frames_req,
                                     target_device_idx=target_device_idx)
        slopec_fsm.inputs['in_pixels'].set(pixels)

        for i in range(1, lock_frames_req + 2):
            pixels.generation_time = t * i
            slopec_fsm.check_ready(t * i)
            slopec_fsm.trigger()
            slopec_fsm.post_trigger()

        self.assertTrue(np.all(cpuArray(slopec_fsm.is_locked)),
                        "FSM failed to achieve lock during testing scale")
        slopes_fsm_x = cpuArray(slopec_fsm.outputs['out_slopes'].xslopes)

        # Cross-compare magnitudes
        np.testing.assert_allclose(slopes_fsm_x, slopes_sh_x, rtol=1e-2, atol=1e-3,
                err_msg="Normalization mismatch between FSM and standard ShSlopec under tracking state!")

    @cpu_and_gpu
    def test_fsm_hold_during_fading_opt_in(self, target_device_idx, xp):
        """`hold_during_fading=True` (opt-in, off by default -- see
        docstring): total signal loss (Track 3) should HOLD the last
        known position instead of extrapolating with the kinematic
        velocity estimate, when explicitly enabled. Explored 2026-09 as a
        candidate fix for closed-loop divergence at low flux; empirically
        made things *worse* in practice (see README.md), so left off by
        default -- this test only verifies the opt-in behaves as
        documented, it is not a recommendation to enable it."""
        subap_npx, t = 32, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        cntrd = (subap_npx - 1) / 2.0

        # max_v raised so the established velocity below isn't clipped away.
        slopec = FsmHybridSlopec(subapdata, hold_during_fading=True, max_v=5.0,
                                 target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Inject: already locked, well clear of max_missed_frames, with an
        # established velocity of +1.0 px/frame (state_x1 - state_x2) and
        # no accumulated Radar confidence (fresh ema_corr).
        slopec.is_locked[:] = True
        slopec.lock_counter[:] = 10
        slopec.miss_counter[:] = 0
        locked_x = cntrd + 3.0
        slopec.state_x1[:] = locked_x
        slopec.state_x2[:] = locked_x - 1.0
        slopec.state_y1[:] = cntrd
        slopec.state_y2[:] = cntrd

        # Plain background frame: no signal anywhere, so both the
        # instantaneous Sniper and the (freshly-zeroed) Radar report
        # SNR ~= 0 -- a clean Track 3 (total fading) hit.
        bg_frame = xp.full(ccd_shape, 1.0, dtype=xp.float32)
        pixels.pixels = bg_frame
        pixels.generation_time = t
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        self.assertTrue(np.all(cpuArray(slopec.is_locked)),
                        "Should still be locked after a single missed frame")

        new_x = cpuArray(slopec.state_x1)
        np.testing.assert_allclose(new_x, locked_x, atol=1e-6,
            err_msg="Fading (Track 3) extrapolated position instead of holding")

    @cpu_and_gpu
    def test_fsm_hold_during_flicker_opt_in(self, target_device_idx, xp):
        """`hold_during_flicker=True` (opt-in, off by default -- see
        docstring): the EMA ("Radar") fallback track, while tracking,
        should HOLD the last known position rather than re-centroid on
        the temporally-smoothed pixel buffer, when explicitly enabled.
        Explored 2026-09 as a candidate fix for an un-tuned phase-lag
        source; empirically made closed-loop divergence *worse* in
        practice (see README.md), so left off by default -- this test
        only verifies the opt-in behaves as documented, it is not a
        recommendation to enable it."""
        subap_npx, t = 32, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        cntrd = (subap_npx - 1) / 2.0
        n_subaps = subapdata.n_subaps

        slopec = FsmHybridSlopec(subapdata, hold_during_flicker=True,
                                 target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Inject: already locked and stationary (zero velocity, isolating
        # this test from fix #1), no missed frames yet.
        slopec.is_locked[:] = True
        slopec.lock_counter[:] = 10
        slopec.miss_counter[:] = 0
        slopec.state_x1[:] = cntrd
        slopec.state_x2[:] = cntrd
        slopec.state_y1[:] = cntrd
        slopec.state_y2[:] = cntrd

        # Manually craft a strong, unambiguous Radar (ema_corr) peak far
        # from the locked position, standing in for several frames' worth
        # of accumulated EMA confidence -- guarantees radar_yes=True this
        # frame regardless of the instantaneous frame's own content.
        far_x = cntrd + 5.0
        ema_peak = slopec._generate_gaussian(
            xp.full(n_subaps, far_x, dtype=slopec.dtype),
            xp.full(n_subaps, cntrd, dtype=slopec.dtype),
            slopec.fwhm_pix) * 100.0
        slopec.ema_corr = ema_peak
        # ema_pixels stays at its zero-init default -- what the old code
        # would have centroided on for the state update this fix removes.

        # Plain background frame: the instantaneous Sniper sees nothing
        # (SNR ~= 0), so only the Radar (Track 2 / flicker) can fire.
        bg_frame = xp.full(ccd_shape, 1.0, dtype=xp.float32)
        pixels.pixels = bg_frame
        pixels.generation_time = t
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        self.assertTrue(np.all(cpuArray(slopec.is_locked)),
                        "Should still be locked after a single flicker frame")

        new_x = cpuArray(slopec.state_x1)
        np.testing.assert_allclose(new_x, cntrd, atol=1e-6,
            err_msg="Flicker (Track 2) re-centroided on EMA pixels instead of holding")

    @cpu_and_gpu
    def test_fsm_velocity_estimate_smoothing_opt_in(self, target_device_idx, xp):
        """`vel_ema_alpha<1.0` (opt-in, default is 1.0 = no smoothing --
        see docstring): the kinematic velocity estimate becomes a
        smoothed (EMA) quantity instead of the raw one-frame finite
        difference of the locked position, when explicitly enabled.
        Explored 2026-09 as a candidate fix (the raw estimate amplifies
        centroid noise directly for a single sub-aperture, as in the
        MORFEO LO NGS case, with no cross-subaperture median to fall back
        on) but, combined with the hold_during_* options, made closed-loop
        divergence *worse* in practice (see README.md) -- this test only
        verifies the opt-in behaves as documented, it is not a
        recommendation to enable it."""
        subap_npx, t = 32, int(1e9)
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx)
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        cntrd = (subap_npx - 1) / 2.0

        vel_ema_alpha = 0.25
        slopec = FsmHybridSlopec(subapdata, vel_ema_alpha=vel_ema_alpha, max_v=5.0,
                                 target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)

        # Inject an established raw one-frame velocity of +2.0 px/frame,
        # starting from a fresh (zero) smoothed-velocity state.
        slopec.is_locked[:] = True
        slopec.lock_counter[:] = 10
        slopec.miss_counter[:] = 0
        slopec.state_x1[:] = cntrd + 2.0
        slopec.state_x2[:] = cntrd
        slopec.state_y1[:] = cntrd
        slopec.state_y2[:] = cntrd
        self.assertEqual(slopec.vx_smooth, 0.0)

        good_frame = self.generate_spots(ccd_shape, subapdata, xp, shift_dx=0.0)
        pixels.pixels = good_frame
        pixels.generation_time = t
        slopec.check_ready(t)
        slopec.trigger()
        slopec.post_trigger()

        expected = vel_ema_alpha * 2.0
        np.testing.assert_allclose(cpuArray(slopec.vx_smooth), expected, atol=1e-6,
            err_msg="Velocity estimate jumped to the raw value instead of being smoothed")

    @cpu_and_gpu
    def test_fsm_defaults_match_original_behaviour(self, target_device_idx, xp):
        """Guards the 2026-09 restoration: with no options passed,
        `hold_during_fading`/`hold_during_flicker` default to `False` and
        `vel_ema_alpha` defaults to 1.0 (no smoothing) -- i.e. the class's
        original, empirically-better behaviour (see README.md), with the
        three candidate fixes available strictly opt-in."""
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, 32)
        slopec = FsmHybridSlopec(subapdata, target_device_idx=target_device_idx)
        self.assertFalse(slopec.hold_during_fading)
        self.assertFalse(slopec.hold_during_flicker)
        self.assertEqual(slopec.vel_ema_alpha, 1.0)
