
from specula.processing_objects.iir_filter import IIRFilter
from specula.data_objects.iir_filter_data import IIRFilterData

    
class LowPassFilter(IIRFilter):
    def __init__(self, cutoff_freq, amplif_fact, time_step, delay=0, offset=None, og_shaper=None,                 
                target_device_idx=None, 
                precision=None
                ):
        
        samp_freq = 1 / time_step
        
        iir_filter_data = IIRFilterData.from_fc_and_ampl(cutoff_freq, amplif_fact,
                                               samp_freq, target_device_idx=target_device_idx)

        # Initialize IIRFilter object
        super().__init__(iir_filter_data, delay=delay, offset=offset, og_shaper=og_shaper,
                         target_device_idx=target_device_idx, precision=precision)
