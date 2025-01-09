
import specula
specula.init(0)  # Default target device

import unittest

from specula import cpuArray
from specula.processing_objects.base_operation import BaseOperation
from specula.base_value import BaseValue

from test.specula_testlib import cpu_and_gpu

class TestBaseOperation(unittest.TestCase):

    @cpu_and_gpu
    def test_sum(self, target_device_idx, xp):
        
        value1 = BaseValue(value=1, target_device_idx=target_device_idx)
        value2 = BaseValue(value=2, target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(sum=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup(1, 1)
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 3

    @cpu_and_gpu
    def test_sub(self, target_device_idx, xp):
        
        value1 = BaseValue(value=1, target_device_idx=target_device_idx)
        value2 = BaseValue(value=2, target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(sub=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup(1, 1)
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == -1

    @cpu_and_gpu
    def test_mul(self, target_device_idx, xp):
        
        value1 = BaseValue(value=2, target_device_idx=target_device_idx)
        value2 = BaseValue(value=3, target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(mul=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup(1, 1)
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 6

    @cpu_and_gpu
    def test_div(self, target_device_idx, xp):
        
        value1 = BaseValue(value=6.0, target_device_idx=target_device_idx)
        value2 = BaseValue(value=3.0, target_device_idx=target_device_idx)
        value1.generation_time = 1
        value2.generation_time = 1
        
        op = BaseOperation(div=True, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        op.inputs['in_value2'].set(value2)
        
        op.setup(1, 1)
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 2

    @cpu_and_gpu
    def test_const_sum(self, target_device_idx, xp):
        
        value1 = BaseValue(value=6, target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        op = BaseOperation(constant_sum=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        
        op.setup(1, 1)
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 8

    @cpu_and_gpu
    def test_const_sub(self, target_device_idx, xp):
        
        value1 = BaseValue(value=6, target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        op = BaseOperation(constant_sub=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        
        op.setup(1, 1)
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 4

    @cpu_and_gpu
    def test_const_mul(self, target_device_idx, xp):
        
        value1 = BaseValue(value=6, target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        op = BaseOperation(constant_mul=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        
        op.setup(1, 1)
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 12
        
    @cpu_and_gpu
    def test_const_div(self, target_device_idx, xp):
        
        value1 = BaseValue(value=6.0, target_device_idx=target_device_idx)
        value1.generation_time = 1
        
        op = BaseOperation(constant_div=2, target_device_idx=target_device_idx)
        op.inputs['in_value1'].set(value1)
        
        op.setup(1, 1)
        op.check_ready(1)
        op.prepare_trigger(1)
        op.trigger()
        op.post_trigger()

        assert cpuArray(op.outputs['out_value'].value) == 3

if __name__ == '__main__':
    unittest.main()