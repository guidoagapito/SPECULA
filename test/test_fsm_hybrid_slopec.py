import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.data_objects.pixels import Pixels
from specula.data_objects.subap_data import SubapData
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

    def generate_spots(self, ccd_shape, subapdata, xp, fwhm=1.5, flux=100.0, bg=1.0):
        """
        Generates a dummy CCD with perfect Gaussian spots at the center 
        of each sub-aperture to simulate a high SNR environment.
        """
        np_sub = subapdata.np_sub
        n_subaps = subapdata.n_subaps
        
        ccd = np.full(ccd_shape, bg, dtype=np.float32)
        cntrd = (np_sub - 1) / 2.0
        
        x = np.arange(np_sub) - cntrd
        y = np.arange(np_sub) - cntrd
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

        return xp.asarray(ccd)

    @cpu_and_gpu
    def test_fsm_acquisition_and_tracking(self, target_device_idx, xp):
        """
        Verifies the Finite State Machine (FSM) transition from Acquisition to Tracking.
        """
        lock_frames_req = 3
        subap_npx = 32
        t = int(1e9)
        
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx=subap_npx)
        
        # Initialize the Slopec object
        slopec = FsmHybridSlopec(subapdata, 
                                 fwhm_pix=1.5, 
                                 snr_thr=3.5, 
                                 lock_frames_req=lock_frames_req,
                                 target_device_idx=target_device_idx)
        
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        
        # Create a frame with perfect spots (very high SNR)
        good_frame = self.generate_spots(ccd_shape, subapdata, xp)
        
        # --- TEST 1: Bootstrap (Acquisition Phase) ---
        for i in range(1, lock_frames_req + 1):
            pixels.pixels = good_frame
            pixels.generation_time = t * i
            
            slopec.check_ready(t * i)
            slopec.trigger()
            slopec.post_trigger()
            
            is_locked_cpu = cpuArray(slopec.is_locked)
            lock_counter_cpu = cpuArray(slopec.lock_counter)
            
            if i < lock_frames_req:
                # We shouldn't be locked yet
                self.assertFalse(np.all(is_locked_cpu), f"Locked too early at frame {i}")
                self.assertTrue(np.all(lock_counter_cpu == i), "Lock counter not incremented correctly")
            else:
                # At frame `lock_frames_req`, the FSM must switch to True
                self.assertTrue(np.all(is_locked_cpu), "FSM did not switch to Tracking Mode")
                
        # Ensure the slopes (measured relative to the center) are zero
        slopes = cpuArray(slopec.outputs['out_slopes'].slopes)
        np.testing.assert_allclose(slopes, 0.0, atol=1e-3, err_msg="Centroids are not at the nominal center")

    @cpu_and_gpu
    def test_fsm_hold_and_miss_counter(self, target_device_idx, xp):
        """
        Verifies that an empty frame does not break the loop instantly, 
        but increments the miss_counter sending the sub-aperture into Hold.
        """
        lock_frames_req = 2
        max_missed_frames = 5
        subap_npx = 32
        t = int(1e9)
        
        subapdata, ccd_shape = self.get_test_setup(target_device_idx, xp, subap_npx=subap_npx)
        
        slopec = FsmHybridSlopec(subapdata, 
                                 snr_thr=3.5, 
                                 lock_frames_req=lock_frames_req,
                                 max_missed_frames=max_missed_frames,
                                 target_device_idx=target_device_idx)
        
        pixels = Pixels(*ccd_shape, target_device_idx=target_device_idx)
        slopec.inputs['in_pixels'].set(pixels)
        
        good_frame = self.generate_spots(ccd_shape, subapdata, xp)
        bad_frame = xp.full(ccd_shape, 1.0, dtype=xp.float32) # Only constant background
        
        # 1. Force the Lock
        for i in range(1, lock_frames_req + 1):
            pixels.pixels = good_frame
            pixels.generation_time = t * i
            slopec.check_ready(t * i)
            slopec.trigger()
            slopec.post_trigger()
            
        self.assertTrue(np.all(cpuArray(slopec.is_locked)), "Setup failed: system not locked")
        
        # 2. Insert an empty frame (Cosmic Ray / Star Fading simulation)
        pixels.pixels = bad_frame
        pixels.generation_time = t * (lock_frames_req + 1)
        slopec.check_ready(pixels.generation_time)
        slopec.trigger()
        slopec.post_trigger()
        
        # HOLD state verifications
        is_locked_cpu = cpuArray(slopec.is_locked)
        miss_counter_cpu = cpuArray(slopec.miss_counter)
        
        self.assertTrue(np.all(is_locked_cpu), "The system lost lock immediately (No hysteresis!)")
        self.assertTrue(np.all(miss_counter_cpu == 1), "miss_counter did not increment on the empty frame")
        
        # Slopes must remain frozen at the last valid position (0.0 in our case)
        slopes = cpuArray(slopec.outputs['out_slopes'].slopes)
        np.testing.assert_allclose(slopes, 0.0, atol=1e-3, err_msg="Positions were not frozen in Hold mode")
