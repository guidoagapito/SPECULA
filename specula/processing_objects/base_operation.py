from astropy.io import fits

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue


def convert_to_xp_array(obj, xp, dtype):
    '''Convert scalar values like 2.0 to xp arrays'''
    if obj:
        value = obj.value
        if xp.isscalar(value):
            v1 = xp.zeros(1, dtype=xp.array(value).dtype) + value
        else:
            v1 = value
    else:
        v1 = xp.zeros(1, dtype=dtype)
    return v1


class BaseOperation(BaseProcessingObj):
    ''''Simple operations with base value(s)'''

    def __init__(self, constant_mul=None, constant_div=None, constant_sum=None, constant_sub=None, mul=False, div=False, sum=False, sub=False,
                 concat=False, target_device_idx=None, precision=None):
        """
        Initialize the base operation object.

        Parameters:
        constant_mul (float, optional): Constant for multiplication
        constant_div (float, optional): Constant for division
        constant_sum (float, optional): Constant for addition
        constant_sub (float, optional): Constant for subtraction
        mul (bool, optional): Flag for multiplication operation
        div (bool, optional): Flag for division operation
        sum (bool, optional): Flag for addition operation
        sub (bool, optional): Flag for subtraction operation
        concat (bool, optional): Flag for concatenation operation
        """
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Implement constant div and sub as reciprocal of mul and sum
        self.constant_mul = constant_mul
        self.constant_sum = constant_sum
        if constant_div is not None:
            self.constant_mul = 1.0 / constant_div
        if constant_sub is not None:
            self.constant_sum = -constant_sub

        self.mul = mul
        self.div = div
        self.sum = sum
        self.sub = sub
        self.concat = concat
        self.out_value = BaseValue(target_device_idx=target_device_idx)

        self.inputs['in_value1'] = InputValue(type=BaseValue)
        self.inputs['in_value2'] = InputValue(type=BaseValue, optional=True)
        self.outputs['out_value'] = self.out_value
        
    def trigger_code(self):
        value1 = self.local_inputs['in_value1'].value
        out = self.out_value.value

        if self.constant_mul:
            out[:] = value1 * self.constant_mul
            return
        if self.constant_sum:
            out[:] = value1 + self.constant_sum
            return

        value2 = self.local_inputs['in_value2'].value

        if self.mul:
            out[:] = value1
            out *= value2
        elif self.div:
            out[:] = value1
            out /= value2
        elif self.sum:
            out[:] = value1
            out += value2
        elif self.sub:
            out[:] = value1
            out -= value2
        elif self.concat:
            out[:len(value1)] = value1
            out[len(value1):] = value2
        else:
            raise ValueError('No operation defined')

    def post_trigger(self):
        super().post_trigger()
        self.out_value.generation_time = self.current_time

    def setup(self, loop_dt, loop_niters):
        '''
        Convert eventual input scalars to self.xp arrays
        and preallocate output value
        
        TODO: the scalar-to-array conversion for inputs
        could be done in prepare_trigger() by the base class
        or even better in InputValue.get() while transferring inputs
        '''
        super().setup(loop_dt, loop_niters)

        value1 = self.inputs['in_value1'].get(target_device_idx=self.target_device_idx)
        value2 = self.inputs['in_value2'].get(target_device_idx=self.target_device_idx)
        
        v1 = convert_to_xp_array(value1, self.xp, self.dtype)
        v2 = convert_to_xp_array(value2, self.xp, self.dtype)

        # Pre-allocate output value
        if self.concat:
            self.out_value.value = self.xp.concatenate((v1, v2))
        else:
            self.out_value.value = v1.copy()


