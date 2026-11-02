from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.data_objects.intensity import Intensity


class IntensityGate(BaseProcessingObj):
    """
    Multiplies a sensor intensity by a time-varying gain (test utility).

    Placed between a wavefront sensor and its detector, with the gain driven by a
    :class:`TimeHistoryGenerator`, it reproduces star dropouts (gain 0) and thin-cloud fades
    (0 < gain < 1) while the detector noise (readout, dark, background) is left untouched.
    """

    def __init__(self,
                 dimx: int = 240,
                 dimy: int = 240,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.out_i = Intensity(dimx, dimy, target_device_idx=self.target_device_idx, precision=precision)
        self.inputs['in_i'] = InputValue(type=Intensity)
        self.inputs['in_gain'] = InputValue(type=BaseValue)
        self.outputs['out_i'] = self.out_i

    @classmethod
    def input_names(cls):
        return {'in_i': InputDesc(Intensity, 'Input intensity'),
                'in_gain': InputDesc(BaseValue, 'Multiplicative gain (first element is used)')}

    @classmethod
    def output_names(cls):
        return {'out_i': OutputDesc(Intensity, 'Gated intensity')}

    def trigger_code(self):
        gain = self.local_inputs['in_gain'].value[0]
        self.out_i.i[:] = self.local_inputs['in_i'].i * gain

    def post_trigger(self):
        super().post_trigger()
        self.out_i.generation_time = self.current_time
