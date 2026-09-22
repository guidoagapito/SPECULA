import specula
specula.init(0)  # Default target device

import unittest

from specula import np
from specula import cpuArray

from specula.base_value import BaseValue
from specula.data_objects.intensity import Intensity
from specula.processing_objects.intensity_gate import IntensityGate
from test.specula_testlib import cpu_and_gpu


class TestIntensityGate(unittest.TestCase):

    @cpu_and_gpu
    def test_gain_scales_the_intensity(self, target_device_idx, xp):
        gate = IntensityGate(dimx=8, dimy=6, target_device_idx=target_device_idx)
        ref = np.arange(48, dtype=float).reshape(6, 8) + 1.0
        for t, gain in enumerate([1.0, 0.0, 0.3], start=1):
            i = Intensity(8, 6, target_device_idx=target_device_idx)
            i.i[:] = xp.asarray(ref)
            i.generation_time = t
            g = BaseValue(value=xp.asarray([gain]), target_device_idx=target_device_idx)
            g.generation_time = t
            gate.inputs['in_i'].set(i)
            gate.inputs['in_gain'].set(g)
            gate.check_ready(t)
            gate.trigger()
            gate.post_trigger()
            np.testing.assert_allclose(cpuArray(gate.outputs['out_i'].i), ref * gain, rtol=1e-6)
            self.assertEqual(gate.outputs['out_i'].generation_time, t)

    def test_names(self):
        self.assertIn('in_gain', IntensityGate.input_names())
        self.assertIn('out_i', IntensityGate.output_names())


if __name__ == '__main__':
    unittest.main()
