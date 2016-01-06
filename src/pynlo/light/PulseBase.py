# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 13:48:11 2015
This file is part of pyNLO.

    pyNLO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    pyNLO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with pyNLO.  If not, see <http://www.gnu.org/licenses/>.
@author: ycasg
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import constants, signal 
from pynlo.util import FFT_t, IFFT_t
import exceptions
import warnings

class Pulse:
    """Class which carried all information about the light field. This class 
       is a base upon which various cases are built (eg analytic pulses,
       CW fields, or pulses generated from experimental data.) """

    def __init__(self, frep_MHz = None, n = None, external_units = None):
        if frep_MHz is not None:
            self._frep_MHz = frep_MHz
            if frep_MHz > 1.0e6:
                warnings.warn("frep should be specified in MHz; large value given.")
        if n is not None:
            self.set_NPTS(n)
            
        if external_units is not None:
            if (external_units == 'mks' or external_units == 'nmps'):
                self.set_units(external_units)        
            else:
                raise exceptions.ValueError('External units must be mks or nmps')

        
    # Private variables:
    # This set is the minimum number required to completely specify the light
    # field. All other representations are derived from them.
    _n                  = 0       # Number of points on grid
    _centerfrequency    = 1.0     # Center frequency (THz)
    _time_window        = 1.0     # Time window (ps)
    _V                  = None    # Relative angular frequency grid (2 pi THz)
    _AW                 = None    # Frequency-domain pulse amplitude
    _frep_MHz           = 100.0   # Pulse frequency (MHz); used for converting
                                  # pulse energy < - > average power
    _ready              = False   # All fields are initialized (this allows for
                                  # incomplete Pulse objects to be created and
                                  # and filled in later)         
    _external_units     = None
    # Constants
    _c_nmps = constants.speed_of_light*1e9/1e12 # c in nm/ps
    _c_mks  = constants.speed_of_light # m/s
    # Cached values for expensive functions that I have identified as widely-used
    # in a profiler. Note that this is a sparse list...
    # Wavelength
    _cache_wl_nm_hash               = None
    _cache_wl_nm                    = None
    # Frequency in THz
    _cache_W_Hz_hash                = None
    _cache_W_Hz                     = None
    
    _not_ready_msg = 'Pulse class is not yet ready -- set center wavelength, time window, and npts.'    
    ####### Private properties    #############################################
    def __get_w0(self):
        if self._centerfrequency is None:
            raise exceptions.ValueError('Center frequency is not set.')
        return 2.0 * np.pi * self._centerfrequency    
    def __get_W(self):
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        else:
            return self._V + self._w0
    def __get_T(self):
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        else:
            TGRID =  np.linspace(-self._time_window / 2.0,
                                  self._time_window / 2.0,
                                  self._n, endpoint = False) # time grid
            return TGRID
    def __get_dT(self):
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        else:
            return self._time_window / np.double(self._n)

    def __get_V(self):
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        else:
            VGRID = 2.0*np.pi*np.transpose(np.arange(-self._n/2,
                                                      self._n/2))/(self._n*self._dT) # Frequency grid (angular THz)        
            return VGRID

            
    _w0     = property(__get_w0)        
    _W      = property(__get_W)
    _dT     = property(__get_dT)                          
    _T      = property(__get_T)
    _dT     = property(__get_dT)
    _V      = property(__get_V)

    ####### Public  properties    #############################################    
    # The following are properties. They should be used by other codes to
    # provide some isolation from the Pulse class's internal mechanics.
    
    # Wavelength is dynamically derived from frequency grid
    def _get_wavelength_nm(self):
        if (self.cache_hash == self._cache_wl_nm_hash):
           return self._cache_wl_nm
        else:
           self._cache_wl_nm_hash = self.cache_hash
           self._cache_wl_nm = 2*np.pi*self._c_nmps / self.W_THz
           return self._cache_wl_nm
    # Wavelength is dynamically derived from frequency grid
    def _get_wavelength_m(self):    
        return 2*np.pi*self._c_mks / self.W_mks
    def _get_center_wavelength_nm(self):
        return self._c_nmps / self._centerfrequency
    def _get_center_wavelength_mks(self):
        return (self._c_nmps / self._centerfrequency )*1.0e9
    def _get_center_frequency_THz(self):
        return self._centerfrequency        
    def _get_center_frequency_mks(self):
        return self._centerfrequency * 1.0e12
        
    def _get_NPTS(self):
        return self._n
    def _get_hash(self):
        return str(self._centerfrequency)+str(self.NPTS)    
    ####### Time                  #############################################            
    def _get_W_Hz(self):
        if (self._cache_W_Hz_hash == self.cache_hash):
            return self._cache_W_Hz
        else:
            self._cache_W_Hz_hash = self.cache_hash
            self._cache_W_Hz                 =  self._W * 1e12
            return self._cache_W_Hz
            
    def _get_W_THz(self):
        return self._W
    def _get_dT_seconds(self):
        return self._dT * 1e-12
    def _get_dT_picoseconds(self):
        return self._dT
        
    def _get_T_seconds(self):
        return self._T* 1e-12
    def _get_T_picoseconds(self):
        return self._T

    def _get_time_window_seconds(self):
        return self._time_window* 1e-12
    def _get_time_window_picoseconds(self):
        return self._time_window

    def _get_V_Hz(self):
        return self._V* 1e12
    def _get_V_THz(self):
        return self._V
    def _get_dF_THz(self):
        return abs(self.W_THz[1]-self.W_THz[0])/(2.0*np.pi)
    def _get_dF_Hz(self):
        return abs(self.W_mks[1]-self.W_mks[0])/(2.0*np.pi)
    def _get_frep_MHz(self):
        return self._frep_MHz
    def _get_frep_Hz(self):
        if self._frep_MHz is None:
            return None
        else:
            return self._frep_MHz * 1.0e6

    ####### ELectrtic Field       #############################################            
    def _get_AW(self):
        if self._AW is not None:
            return self._AW.copy()
        else:
            raise exceptions.RuntimeError('Grids not yet set up.')
    def _get_AT(self):        
        return IFFT_t( self._AW.copy() )

    def set_AW(self, AW_new):
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        if self._AW is None:
            self._AW = np.zeros((self._n,), dtype = np.complex128)
        self._AW[:] = AW_new
        
    def set_AT(self, AT_new):
        self.set_AW( FFT_t(AT_new ))

    # To keep this class' working isolated from accessors, all data reading and
    # writing is done via methods. These are:

    wl_nm           = property(_get_wavelength_nm)
    W_THz           = property(_get_W_THz)
    dT_ps           = property(_get_dT_picoseconds)
    T_ps            = property(_get_T_picoseconds)
    V_THz           = property(_get_V_THz)
    time_window_ps  = property(_get_time_window_picoseconds)
    center_wavelength_nm    = property(_get_center_wavelength_nm)
    center_frequency_THz = property(_get_center_frequency_THz)
    
    wl_mks          = property(_get_wavelength_m)
    W_mks           = property(_get_W_Hz)      
    dT_mks          = property(_get_dT_seconds)
    T_mks           = property(_get_T_seconds)
    V_mks           = property(_get_V_Hz)
    dF_mks           = property(_get_dF_Hz)    
    dF_THz           = property(_get_dF_THz)
    
    time_window_mks = property(_get_time_window_seconds)
    center_wavelength_mks   = property(_get_center_wavelength_mks)
    center_frequency_mks = property(_get_center_frequency_mks)
    
    
    AW              = property(_get_AW)
    AT              = property(_get_AT)
    NPTS            = property(_get_NPTS)  
    frep_MHz        = property(_get_frep_MHz)
    frep_Hz         = property(_get_frep_Hz)

    cache_hash      = property(_get_hash)
    
    def _set_centerfrequency(self, f_THz):
        self._centerfrequency = f_THz
        self._check_ready()
    def _set_time_window(self, T_ps):
        self._time_window = T_ps
        self._check_ready()

    def _check_ready(self):
        self._ready =  (self._centerfrequency is not None) and\
                       (self._n is not None) and\
                       (self._time_window is not None)
               

    def _ext_units_nmps(self):
        if self._external_units is None:
            exceptions.RuntimeError('Unit type has not been set.')
        return self._external_units == 'nmps'
    def _ext_units_mks(self):
        if self._external_units is None:
            exceptions.RuntimeError('Unit type has not been set.')
        return self._external_units == 'mks'

    ####### Core public  functions     ########################################        
    def set_center_wavelength_nm(self, wl):
        self._set_centerfrequency(self._c_nmps / wl)
    def set_center_wavelength_m(self, wl):
        self._set_centerfrequency(self._c_nmps /  (wl * 1e9) )
    def set_NPTS(self, NPTS):
        self._n = int(NPTS)
        self._check_ready() 
    def set_frep_MHz(self, fr_MHz):
        self._frep_MHz = fr_MHz
    def set_time_window_ps(self, T):
        if self._n is None:
            raise exceptions.RuntimeError('Set number of points before setting time window.')
        # frequency grid is 2 pi/ dT * [-1/2, 1/2]
        # dT is simply time_window / NPTS
        self._set_time_window(T)
    def set_time_window_s(self, T):
        if self._n is None:
            raise exceptions.RuntimeError('Set number of points before setting time window.')        
        self._set_time_window(T * 1e12)
        
    def set_frequency_window_THz(self, DF):
        if self._n is None:
            raise exceptions.RuntimeError('Set number of points before setting frequency window.')
        # Internally, the time window is used to determine the grids. Calculate
        # the time window size as  1 / dF = 1 / (DF / N)
        T = self._n / float(DF)
        self._set_time_window(T)
    def set_frequency_window_Hz(self, DF):
        if self._n is None:
            raise exceptions.RuntimeError('Set number of points before setting frequency window.')
        # Internally, the time window is used to determine the grids. Calculate
        # the time window size as  1 / dF = 1 / (DF / N)
        T = self._n / float(DF)
        self._set_time_window(T * 1e12)
        

    
    def set_units(self, external_units):
        if external_units == 'nmps':
            self.external_c = self._c_nmps
            self._external_units = external_units                    
        elif external_units == 'mks':
            self.external_c = self._c_mks
            self._external_units = external_units        
        else:
            exceptions.ValueError('Unit type ',external_units,' is not known. Valid values are nmps and mks.')    
    


        
    def internal_time_from_ps(self, time, power = 1):        
        """ Convert to internal units of ps"""
        if self._ext_units_nmps():
            return time
        if self._ext_units_mks():
            return time * (1e-12)**power
    def internal_time_to_ps(self, time, power = 1):
        """ Convert from internal units of ps to external time """
        if self._ext_units_nmps():
            return time
        if self._ext_units_mks():
            return time * (1e12)**power                    
            
    def internal_wl_from_nm(self, wl):
        """ Convert to internal units of nm """
        if self._ext_units_nmps():
            return wl
        if self._ext_units_mks():
            return wl * 1e-9
    def internal_wl_to_nm(self, wl):
        """ Convert from internal units of nm to external units """
        if self._ext_units_nmps():
            return wl
        if self._ext_units_mks():
            return wl * 1e9
                

    ####### Auxiliary public  functions     ###################################
    def calc_epp(self):
        ''' Calculate and return energy per pulse via numerical integration
            of A^2 dt'''
        return self.dT_mks * np.trapz(abs(self.AT)**2)
        
    def chirp_pulse_W(self, GDD, TOD, FOD = 0.0, w0_THz = None):
        ''' Add GDD and TOD to the pulse.'''
        if w0_THz is None:
            self.set_AW( np.exp(1j * (GDD / 2.0) * self.V_THz**2 + 
                                   1j * (TOD / 6.0) * self.V_THz**3+ 
                                   1j * (FOD / 24.0) * self.V_THz**4) * self.AW )
        else:
            V = self.W_THz - w0_THz
            self.set_AW( np.exp(1j * (GDD / 2.0) * V**2 + 
                                   1j * (TOD / 6.0) * V**3+ 
                                   1j * (FOD / 24.0) * V**4) * self.AW )
    def apply_phase_W(self, phase):
        self.set_AW(self.AW * np.exp(1j*phase))
    def chirp_pulse_T(self, chirp2, chirp3, T0):
        self.set_AT( self.AT * np.exp(-1j * (chirp2 / 2.0) * (self.T_ps/T0)**2 + 
                                 -1j * (chirp3 / 3.0) * (self.T_ps/T0)**3) )
                                 
    def dechirp_pulse(self, GDD_TOD_ratio = 0.0, intensity_threshold = 0.05):

        spect_w = self.AW
        phase   = np.unwrap(np.angle(spect_w))
        ampl    = np.abs(spect_w)
        mask = ampl**2 > intensity_threshold * np.max(ampl)**2
        gdd     = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 2))
#        plt.figure()
#        plt.plot(self.W[mask], phase[mask])
#        plt.plot(self.W[mask], phase[mask]-gdd(self.W[mask]))
#        plt.show()
        self.set_AW( ampl * np.exp(1j*(phase-gdd(self.W_THz))) )
    def remove_time_delay(self, intensity_threshold = 0.05):

        spect_w = self.AW
        phase   = np.unwrap(np.angle(spect_w))
        ampl    = np.abs(spect_w)
        mask = ampl**2 > (intensity_threshold * np.max(ampl)**2)        
        ld     = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 1))
#        plt.figure()
#        plt.plot(self.W[mask], phase[mask])
#        plt.plot(self.W[mask], phase[mask]-gdd(self.W[mask]))
#        plt.show()
        self.set_AW( ampl * np.exp(1j*(phase-ld(self.W_THz))) )
    def add_time_offset(self, offset_ps):
        """Shift field in time domain by offset_ps picoseconds. A positive offset
           moves the pulse forward in time. """
        phase_ramp = np.exp(-1j*self.W_THz*offset_ps)
        self.set_AW(self.AW * phase_ramp)
    def rotate_spectrum_to_new_center_wl(self, new_center_wl_nm):
        """Change center wavelength of pulse by rotating the electric field in
            the frequency domain. Designed for creating multiple pulses with same
            gridding but of different colors. Rotations is by integer and to
            the closest omega."""
        new_center_THz = self._c_nmps/new_center_wl_nm
        rotation = (self.center_frequency_THz-new_center_THz)/self.dF_THz
        self.set_AW(np.roll(self.AW, int(round(rotation))))
    def filter_by_wavelength_nm(self, lower_wl_nm, upper_wl_nm):
        AW_new = self.AW
        AW_new[self.wl_nm < lower_wl_nm] = 0.0
        AW_new[self.wl_nm > upper_wl_nm] = 0.0
        self.set_AW(AW_new)
        
    def clone_pulse(self, p):
        '''Copy all parameters of pulse_instance into this one'''
        self.set_NPTS(p.NPTS)
        self.set_time_window_ps(p.time_window_ps)
        self.set_center_wavelength_nm(p.center_wavelength_nm)
        self._frep_MHz = p.frep_MHz
        self.set_AT(p.AT)
    def create_cloned_pulse(self):
        '''Create and return new pulse instance identical to this instance.'''
        p = Pulse()
        p.clone_pulse(self)
        return p               
    def create_subset_pulse(self, center_wl_nm, NPTS):
        """ Create new pulse with smaller frequency span, centered at closest 
            grid point to center_wl_nm, with NPTS grid points and
            frequency-grid values from this pulse. """
            
        if NPTS >= self.NPTS:
            raise exceptions.ValueError("New pulse must have fewer points than existing one.")
        p = Pulse()
        center_idx = np.argmin(abs(self.wl_nm - center_wl_nm))

        # We want to reduce the frequency span, which means holding df fixed
        # while reducing NPTS. The time window is 1/df, so this means holding
        # the time window fixed as well.

        p._frep_MHz = self.frep_MHz
        p.set_center_wavelength_nm(self.wl_nm[center_idx] )
        p.set_time_window_ps(self.time_window_ps)
        p.set_NPTS(NPTS)
        idx1 = center_idx - (NPTS >> 1)
        idx2 = center_idx + (NPTS >> 1)
        p.set_AW(self.AW[idx1:idx2])
        return p
        
    def calculate_weighted_avg_frequency_mks(self):
        avg = np.sum(abs(self.AW)**2 * self.W_mks)
        weights = np.sum(abs(self.AW)**2)
        result = avg / (weights * 2.0 * np.pi)
        return result
    def calculate_weighted_avg_wavelength_nm(self):
        return 1.0e9 * self._c_mks / self.calculate_weighted_avg_frequency_mks()
    def calculate_intensity_autocorrelation(self):  
        """ Calculates and returns the intensity autocorrelation,  
        :math:`\int P(t)P(t+{\tau}) dt` """  
        return np.correlate(abs(self.AT)**2, abs(self.AT), mode='same')  
    def write_frog(self,
                 fileloc = 'broadened_er_pulse.dat', # default EDFA spectrum
                 flip_phase = True):
        """Save pulse in FROG data format. Grid is centered at wavelength
        center_wavelength (nm), but pulse properties are loaded from data
        file. If flip_phase is true, all phase is multiplied by -1 [useful
        for correcting direction of time ambiguity]. time_window (ps) sets 
        temporal grid size. 
        
        power sets the pulse energy:
        if power_is_epp is True  then the number is pulse energy [J] 
        if power_is_epp is False then the power is average power [W], and 
        is multiplied by frep to calculate pulse energy"""       
        
        self.fileloc   = fileloc             
        phase_data = np.unwrap(np.angle(self.get_AW()))
        inten_data = np.abs(self.get_AW())
        wavel_data = self.internal_wl_to_nm(self.wl_nm)
        
        # Write pulse data file
        np.savetxt(self.fileloc, np.vstack((wavel_data, inten_data, phase_data)).T) 
