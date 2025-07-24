import numpy as np
from collections import OrderedDict, defaultdict

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.data_objects.electric_field import ElectricField
from specula.data_objects.pixels import Pixels
from specula.data_objects.slopes import Slopes
from specula.data_objects.intensity import Intensity
from specula.connections import InputValue


class DataBuffer(BaseProcessingObj):
    '''Data buffering object - accumulates data and outputs it every N steps'''

    def __init__(self,
                 buffer_size: int = 10,
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        self.buffer_size = buffer_size
        self.storage = defaultdict(OrderedDict)  # Same structure as DataStore
        self.step_counter = 0
        self.buffered_outputs = {}  # Will hold the actual output objects

        # Will be populated dynamically when inputs are connected
        self.input_types = {}

    def setup(self):
        super().setup()

        # Create output objects based on connected inputs
        for input_name, input_obj in self.local_inputs.items():
            if input_obj is not None:
                # Determine the type and create appropriate output
                sample_data = input_obj
                self.input_types[input_name] = type(sample_data)

                # Create output name and object
                output_name = f"{input_name}_buffered"

                if isinstance(sample_data, BaseValue):
                    # For BaseValue, create an array of values
                    self.buffered_outputs[output_name] = BaseValue(
                        target_device_idx=self.target_device_idx
                    )
                elif isinstance(sample_data, Slopes):
                    # For Slopes, we'll output a stacked array
                    self.buffered_outputs[output_name] = BaseValue(
                        target_device_idx=self.target_device_idx
                    )
                elif isinstance(sample_data, Pixels):
                    # For Pixels, we'll output a stacked array  
                    self.buffered_outputs[output_name] = BaseValue(
                        target_device_idx=self.target_device_idx
                    )
                elif isinstance(sample_data, ElectricField):
                    # For ElectricField, we'll output a stacked array
                    self.buffered_outputs[output_name] = BaseValue(
                        target_device_idx=self.target_device_idx
                    )
                elif isinstance(sample_data, Intensity):
                    # For Intensity, we'll output a stacked array
                    self.buffered_outputs[output_name] = BaseValue(
                        target_device_idx=self.target_device_idx
                    )
                else:
                    # Generic case - create BaseValue
                    self.buffered_outputs[output_name] = BaseValue(
                        target_device_idx=self.target_device_idx
                    )
                
                # Add to outputs dictionary
                self.outputs[output_name] = self.buffered_outputs[output_name]

    def trigger_code(self):
        # Accumulate data (same logic as DataStore)
        for k, item in self.local_inputs.items():
            if item is not None and item.generation_time == self.current_time:
                # Extract data using same logic as DataStore
                if isinstance(item, BaseValue):
                    v = cpuArray(item.value, force_copy=True)
                elif isinstance(item, Slopes):
                    v = cpuArray(item.slopes, force_copy=True)
                elif isinstance(item, Pixels):
                    v = cpuArray(item.pixels, force_copy=True)
                elif isinstance(item, ElectricField):
                    v = np.stack((cpuArray(item.A, force_copy=True), 
                                 cpuArray(item.phaseInNm, force_copy=True)))
                elif isinstance(item, Intensity):
                    v = cpuArray(item.i, force_copy=True)
                else:
                    # Generic case - try to extract value
                    if hasattr(item, 'value'):
                        v = cpuArray(item.value, force_copy=True)
                    else:
                        raise TypeError(f"Error: don't know how to buffer an object of type {type(item)}")
                
                self.storage[k][self.current_time] = v
        
        self.step_counter += 1
        
        # Check if buffer is full
        if self.step_counter >= self.buffer_size:
            self.emit_buffered_data()
            self.reset_buffers()

    def emit_buffered_data(self):
        """Emit accumulated data as outputs"""
        for input_name, data_dict in self.storage.items():
            if len(data_dict) == 0:
                continue

            output_name = f"{input_name}_buffered"

            # Stack all accumulated data
            times = np.array(list(data_dict.keys()), dtype=self.dtype)
            values = np.array(list(data_dict.values()))

            # Create output data structure: [times, values]
            output_data = {
                'times': times,
                'data': values,
                'shape': values.shape,
                'input_name': input_name
            }

            # Set the output
            if output_name in self.buffered_outputs:
                self.buffered_outputs[output_name].value = output_data
                self.buffered_outputs[output_name].generation_time = self.current_time

                if self.verbose:
                    print(f"DataBuffer: emitted {len(values)} samples for {input_name}")

    def reset_buffers(self):
        """Clear all buffers and reset counter"""
        self.storage.clear()
        self.step_counter = 0

        if self.verbose:
            print(f"DataBuffer: reset buffers at time {self.current_time}")

    def get_buffer_status(self):
        """Return current buffer fill status"""
        return {
            'step_counter': self.step_counter,
            'buffer_size': self.buffer_size,
            'fill_percentage': (self.step_counter / self.buffer_size) * 100,
            'stored_inputs': list(self.storage.keys())
        }

    def force_emit(self):
        """Force emission of current buffer contents (useful for finalization)"""
        if self.step_counter > 0:
            self.emit_buffered_data()
            self.reset_buffers()

    def finalize(self):
        """Emit any remaining data in buffers"""
        self.force_emit()
        super().finalize()