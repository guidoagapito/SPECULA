import numpy as np
from collections import deque
from specula.base_processing_obj import BaseProcessingObj

class DataBuffer(BaseProcessingObj):
    def __init__(self, name="DataBuffer", buffer_size=10, **kwargs):
        super().__init__(name=name, **kwargs)
        self.buffer_size = buffer_size
        self.buffers = {}
        self.ready_flags = {}

    def set_input_names(self, input_names):
        """Initialize a buffer for each expected input."""
        for name in input_names:
            self.buffers[name] = deque(maxlen=self.buffer_size)
            self.ready_flags[name] = False

    def check_ready(self):
        """Override SPECULA's readiness check."""
        for name, buffer in self.buffers.items():
            if len(buffer) < self.buffer_size:
                return False
        return True

    def prepare_trigger(self, t):
        """No-op unless time-based filtering needed."""
        pass

    def trigger(self):
        if not self.check_ready():
            return

        self.outputs = {}
        for name, buffer in self.buffers.items():
            stacked = np.stack(buffer)
            self.outputs[name + "_buffered"] = stacked
            buffer.clear()  # Clear after output

    def set_inputs(self, input_dict):
        for name, value in input_dict.items():
            if name in self.buffers:
                self.buffers[name].append(np.copy(value))

    def set_output_names(self):
        self.outputs = {}
        for name in self.buffers:
            self.outputs[name + "_buffered"] = None
 