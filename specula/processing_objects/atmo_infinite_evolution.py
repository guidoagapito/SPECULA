import numpy as np
from specula import show_in_profiler

from astropy.io import fits

from specula.base_processing_obj import BaseProcessingObj
from specula.infinite_phase_screen import InfinitePhaseScreen
from specula.base_value import BaseValue
from specula.data_objects.layer import Layer
from specula.lib.cv_coord import cv_coord

from specula.connections import InputValue
from specula import cpuArray, ASEC2RAD

class AtmoInfiniteEvolution(BaseProcessingObj):
    def __init__(self,
                 L0: list,
                 pixel_pitch: float,
                 heights: list,
                 Cn2: list,
                 pixel_pupil: float,
                 data_dir: str,
                 source_dict: dict,
                 wavelengthInNm: float=500.0,
                 zenithAngleInDeg: float=0.0,
                 mcao_fov: float=None,
                 seed: int=1,
                 verbose: bool=False,
                 user_defined_phasescreen: str='',
                 force_mcao_fov: bool=False,
                 fov_in_m: float=None,
                 pupil_position:list =[0,0],
                 target_device_idx: int=None,
                 precision: int=None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)
        
        self.n_infinite_phasescreens = len(heights)
        self.last_position = np.zeros(self.n_infinite_phasescreens)
        self.last_t = 0
        self.delta_time = 1
        # fixed at generation time, then is a input -> rescales the screen?
        self.seeing = 0.8
        self.l0 = 0.005
        self.wind_speed = 1
        self.wind_direction = 1
        self.airmass = 1
        self.ref_wavelengthInNm = wavelengthInNm
        self.pixel_pitch = pixel_pitch         
        
        self.inputs['seeing'] = InputValue(type=BaseValue)
        self.inputs['wind_speed'] = InputValue(type=BaseValue)
        self.inputs['wind_direction'] = InputValue(type=BaseValue)        
        
        self.airmass = 1.0 / np.cos(np.radians(zenithAngleInDeg), dtype=self.dtype)
        # print(f'Atmo_Evolution: zenith angle is defined as: {zenithAngleInDeg} deg')
        # print(f'Atmo_Evolution: airmass is: {self.airmass}')
    
        self.heights = np.array(heights, dtype=self.dtype) * self.airmass

        if force_mcao_fov:
            print(f'\nATTENTION: MCAO FoV is forced to diameter={mcao_fov} arcsec\n')
            alpha_fov = mcao_fov / 2.0
        else:
            alpha_fov = 0.0
            for source in source_dict.values():
                alpha_fov = max(alpha_fov, *abs(cv_coord(from_polar=[source.phi, source.r_arcsec],
                                                       to_rect=True, degrees=False, xp=np)))
            if mcao_fov is not None:
                alpha_fov = max(alpha_fov, mcao_fov / 2.0)
        
        # Max star angle from arcseconds to radians
        rad_alpha_fov = alpha_fov * ASEC2RAD

        # Compute layers dimension in pixels
        self.pixel_layer_size = np.ceil((pixel_pupil + 2 * np.sqrt(np.sum(np.array(pupil_position, dtype=self.dtype) * 2)) / self.pixel_pitch + 
                               2.0 * abs(self.heights) / self.pixel_pitch * rad_alpha_fov) / 2.0) * 2.0
        if fov_in_m is not None:
            self.pixel_layer_size = np.full_like(self.heights, int(fov_in_m / self.pixel_pitch / 2.0) * 2)
        
        self.L0 = L0
        self.Cn2 = np.array(Cn2, dtype=self.dtype)
        self.pixel_pupil = pixel_pupil
        self.data_dir = data_dir
        self.wind_speed = None
        self.wind_direction = None

        self.verbose = verbose if verbose is not None else False
        
        # Initialize layer list with correct heights
        self.layer_list = []
        for i in range(self.n_infinite_phasescreens):
            layer = Layer(self.pixel_layer_size[i], self.pixel_layer_size[i], self.pixel_pitch, self.heights[i], precision=self.precision, target_device_idx=self.target_device_idx)
            self.layer_list.append(layer)
        self.outputs['layer_list'] = self.layer_list
        
        self.initScreens(seed)

        self.last_position = np.zeros(self.n_infinite_phasescreens, dtype=self.dtype)
        
        if not np.isclose(np.sum(self.Cn2), 1.0, atol=1e-6):
            raise ValueError(f' Cn2 total must be 1. Instead is: {np.sum(self.Cn2)}.')

    def initScreens(self, seed):
        self.seed = seed
        if self.seed <= 0:
            raise ValueError('seed must be >0')
        # Phase screens list
        self.infinite_phasescreens = []
        seed = self.seed + self.xp.arange(self.n_infinite_phasescreens)
        if len(seed) != len(self.L0):
            raise ValueError('Number of elements in seed and L0 must be the same!')


        self.acc_rows = np.zeros((self.n_infinite_phasescreens))
        self.acc_cols = np.zeros((self.n_infinite_phasescreens))

        # Square infinite_phasescreens
        for i in range(self.n_infinite_phasescreens):
            print('Creating phase screen..')
            self.ref_r0 = 0.9759 * 0.5 / (self.seeing * 4.848) * self.airmass**(-3./5.) # if seeing > 0 else 0.0
#            self.ref_r0 *= (self.ref_wavelengthInNm / 500.0 / ((2*np.pi)))**(6./5.) 
            self.ref_r0 *= (self.ref_wavelengthInNm / 500.0 )**(6./5.) 
            print('self.ref_r0:', self.ref_r0)
            temp_infinite_screen = InfinitePhaseScreen(self.pixel_layer_size[i], self.pixel_pitch, 
                                                       self.ref_r0,
                                                       self.L0[i], self.l0, xp=self.xp, target_device_idx=self.target_device_idx, precision=self.precision )
            self.infinite_phasescreens.append(temp_infinite_screen)


    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        self.delta_time = self.t_to_seconds(self.current_time - self.last_t)
    
    @show_in_profiler('atmo_evolution.trigger_code')
    def trigger_code(self):
        seeing = cpuArray(self.local_inputs['seeing'].value)
        wind_speed = cpuArray(self.local_inputs['wind_speed'].value)
        wind_direction = cpuArray(self.local_inputs['wind_direction'].value)

        r0 = 0.9759 * 0.5 / (seeing * 4.848) * self.airmass**(-3./5.)
        r0 *= (self.ref_wavelengthInNm / 500)**(6./5.)        
        scale_r0 = (self.ref_r0 / r0)**(5./6.) 

        scale_wvl = ( self.ref_wavelengthInNm / (2 * np.pi) )
        scale_coeff = scale_wvl

#        print('scale_r0', scale_r0)
#        print('scale_coeff', scale_coeff)

        ascreen = scale_coeff * self.infinite_phasescreens[0].scrn
        
        # Compute the delta position in pixels
        delta_position =  wind_speed * self.delta_time / self.pixel_pitch  # [pixel]
        new_position = self.last_position + delta_position
        eps = 1e-4

        for ii, phaseScreen in enumerate(self.infinite_phasescreens):
            w_y_comp = np.cos(2*np.pi*(wind_direction[ii])/360.0)
            w_x_comp = np.sin(2*np.pi*(wind_direction[ii])/360.0)
            frac_rows, rows_to_add = np.modf( delta_position[ii] * w_y_comp + self.acc_rows[ii])            
            #sr = int( (np.sign(rows_to_add) + 1) / 2 )
            sr = int(np.sign(rows_to_add) )
            frac_cols, cols_to_add = np.modf( delta_position[ii] * w_x_comp + self.acc_cols[ii] )
            #sc = int( (-np.sign(cols_to_add) + 1) / 2 )
            sc = int(np.sign(cols_to_add) )
            # print('rows_to_add, cols_to_add', rows_to_add, cols_to_add)            
            if np.abs(w_y_comp)>eps:
                for r in range(int(np.abs(rows_to_add))):
                    phaseScreen.add_line(1, sr)
            if np.abs(w_x_comp)>eps:
                for r in range(int(np.abs(cols_to_add))):
                    phaseScreen.add_line(0, sc)
            phaseScreen0 = phaseScreen.scrnRawAll.copy()
            # print('w_y_comp, w_x_comp', w_y_comp, w_x_comp)
            # print('frac_rows, frac_cols', frac_rows, frac_cols)
            srf = int(np.sign(frac_rows) )
            scf = int(np.sign(frac_cols) )

            if np.abs(frac_rows)>eps:
                phaseScreen.add_line(1, srf, False)
            if np.abs(frac_cols)>eps:
                phaseScreen.add_line(0, scf, False)
            phaseScreen1 = phaseScreen.scrnRawAll.copy()
            interpfactor = np.sqrt(frac_rows**2 + frac_cols**2 )
            layer_phase = interpfactor * phaseScreen1 + (1.0-interpfactor) * phaseScreen0
            phaseScreen.full_scrn = phaseScreen0
            self.acc_rows[ii] = frac_rows
            self.acc_cols[ii] = frac_cols
            # print('acc_rows', self.acc_rows)
            # print('acc_cols', self.acc_cols)
            self.layer_list[ii].phaseInNm = layer_phase * scale_coeff
            self.layer_list[ii].generation_time = self.current_time
        self.last_position = new_position
        self.last_t = self.current_time
        
    def save(self, filename):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['INTRLVD'] = int(self.interleave)
        hdr['PUPD_TAG'] = self.pupdata_tag
        super().save(filename, hdr)

        with fits.open(filename, mode='append') as hdul:
            hdul.append(fits.ImageHDU(data=self.infinite_phasescreens))

    def read(self, filename):
        super().read(filename)
        self.infinite_phasescreens = fits.getdata(filename, ext=1)

    def set_last_position(self, last_position):
        self.last_position = last_position

    def set_last_t(self, last_t):
        self.last_t = last_t



