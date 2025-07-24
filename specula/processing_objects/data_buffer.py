import numpy as np

from collections import OrderedDict, defaultdict

from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.intensity import Intensity

class DataBuffer(BaseProcessingObj):
    '''Data buffering object - accumulates data and outputs it every N steps'''

    def __init__(self, buffer_size: int = 10):
        super().__init__()
        self.buffer_size = buffer_size
        self.storage = defaultdict(OrderedDict)
        self.step_counter = 0
        self.buffered_outputs = {}

    def setup(self):
        super().setup()

        # Create output objects for each input (like DataStore does)
        for input_name, input_obj in self.local_inputs.items():
            if input_obj is not None:
                # Create output name and object
                output_name = f"{input_name}_buffered"
                output_obj = BaseValue(target_device_idx=self.target_device_idx)
                self.buffered_outputs[output_name] = output_obj
                self.outputs[output_name] = output_obj

    def trigger_code(self):
        # Accumulate data (same logic as DataStore)
        for k, item in self.local_inputs.items():
            if item is not None and item.generation_time == self.current_time:
                if isinstance(item, BaseValue):
                    v = item.value
                elif isinstance(item, Slopes):
                    v = item.slopes
                elif isinstance(item, Pixels):
                    v = item.pixels
                elif isinstance(item, ElectricField):
                    v = np.stack((item.A, item.phaseInNm))
                elif isinstance(item, Intensity):
                    v = item.i
                else:
                    raise TypeError(f"Error: don't know how to buffer an object of type {type(item)}")
                self.storage[k][self.current_time] = v

        self.step_counter += 1

        if self.step_counter >= self.buffer_size:
            self.emit_buffered_data()
            self.reset_buffers()

    def emit_buffered_data(self):
        for input_name, data_dict in self.storage.items():
            if len(data_dict) == 0:
                continue
            output_name = f"{input_name}_buffered"
            values = self.xp.array(list(data_dict.values()))
            if output_name in self.buffered_outputs:
                self.buffered_outputs[output_name].value = values
                self.buffered_outputs[output_name].generation_time = self.current_time
                if self.verbose:
                    print(f"DataBuffer: emitted {len(values)} samples for {input_name}")

    def reset_buffers(self):
        """Clear all buffers and reset counter"""
        self.storage.clear()
        self.step_counter = 0

        if self.verbose:
            print(f"DataBuffer: reset buffers at time {self.current_time}")

    def finalize(self):
        """Emit any remaining data in buffers"""
        self.trigger_code()
        if self.step_counter > 0:
            self.emit_buffered_data()
            self.reset_buffers()
        
        super().finalize()