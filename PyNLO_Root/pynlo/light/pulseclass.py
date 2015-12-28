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
from scipy import constants 
from gnlse_ffts import FFT_t, IFFT_t
import exceptions

class Pulse:
    """Class which carried all information about the light field. Includes
    functions for generating standard pulse shapes and also loading complex
    spectra from FROG data. Initialize sets number of points in grid.
    Pulse repetition frequency frep is required to convert from pulse energy to
    average power."""
    # Constants
    c_nmps = constants.speed_of_light*1e9/1e12 # c in nm/ps
    c_mks  = constants.speed_of_light # m/s
    
    # The following are properties. They should be used by other codes to
    # provide some isolation from the Pulse class's internal mechanics.
    
    # Wavelength is dynamically derived from frequency grid
    def _get_wavelength_nm(self):
        if self.external_units == 'nm/ps':
            return 2*np.pi*self.c / self.W
        elif self.external_units == 'mks':
            return 2e9*np.pi*self.c / self.W
        else:
            raise exceptions.AttributeError('Pulse units are incorrect.')
    wl_nm = property(_get_wavelength_nm)

    # Wavelength is dynamically derived from frequency grid
    def _get_wavelength_m(self):
        if self.external_units == 'nm/ps':
            return 2*np.pi*self.c_nmps / self.W_THz
        elif self.external_units == 'mks':
            return 2*np.pi*self.c_mks / self.W_hz
        else:
            raise exceptions.AttributeError('Pulse units are incorrect.')            
    wl_m = property(_get_wavelength_m)
    # Wavelength is dynamically derived from frequency grid
    def _get_NPTS(self):
        return len(self.A)        
    NPTS = property(_get_NPTS)  
    # angular frequency (Hz)
    def _get_W_hz(self):
        return self.W * 1e12
    W_hz = property(_get_W_hz)  
    
    def _get_dT_seconds(self):
        return self.dT * 1e-12
    dt_seconds = property(_get_dT_seconds)
    
    def _get_W_THz(self):
        return self.W
    W_Thz = property(_get_W_THz)  
    
    def __init__(self, frep = 100e6, n = 2**12, external_units = 'nm/ps'):
        self.external_units = external_units
        self.n = n # n points        
        if external_units == 'nm/ps':
            self.external_c = self.c_nmps
        if external_units == 'mks':
            self.external_c = self.c_mks
        
        # Set the average power coupled into the nonlinear fiber
        self.frep = frep        
    
    def internal_time_from_ps(self, time, power = 1):
        """ Convert to internal units of ps"""
        if self.external_units == 'nm/ps':
            return time
        if self.external_units == 'mks':
            return time * (1e-12)**power
    def internal_time_to_ps(self, time, power = 1):
        """ Convert from internal units of ps to external time """
        if self.external_units == 'nm/ps':
            return time
        if self.external_units == 'mks':
            return time * (1e12)**power                    
    def internal_wl_from_nm(self, wl):
        """ Convert to internal units of nm """
        if self.external_units == 'nm/ps':
            return wl
        if self.external_units == 'mks':
            return wl * 1e-9
    def internal_wl_to_nm(self, wl):
        """ Convert from internal units of nm to external units """
        if self.external_units == 'nm/ps':
            return wl
        if self.external_units == 'mks':
            return wl * 1e9
    
    def gen_sech(self, power, T0, center_wavelength, time_window = 10.,
                 GDD = 0, TOD = 0, chirp2 = 0, chirp3 = 0,
                 power_is_avg = False):
        """Generate sech pulse A(t) = sqrt(peak_power [W]) * sech(t/T0 [ps])
        centered at wavelength center_wavelength (nm).
        time_window (ps) sets temporal grid size. Optional GDD and TOD are
        in ps^2 and ps^3."""

        time_window = self.internal_time_from_ps(time_window)
        T0 = self.internal_time_from_ps(T0)
        center_wavelength = self.internal_wl_from_nm(center_wavelength)
        GDD = self.internal_time_from_ps(GDD, 2)    
        TOD = self.internal_time_from_ps(TOD, 3) 
        
        # The free parameters are time_window, wavelength, and grid size.
        self.twidth = time_window          # Time window in ps
        self.center_wl = center_wavelength    # reference wavelength (nm)              

        self.setup_grids()
        
        ### Generate pulse
        if not power_is_avg:
            self.A = np.sqrt(power)/np.cosh(self.T/T0)
        if power_is_avg:
            self.A = 1 / np.cosh(self.T/T0)            
        
        self.chirp_pulse_W(GDD, TOD)
        self.chirp_pulse_T(chirp2, chirp3, T0)            
        
    def gen_gaussian(self, power, T0, center_wavelength, time_window = 10.,
                     GDD = 0, TOD = 0, chirp2 = 0, chirp3 = 0,
                     power_is_avg = False):
        """Generate Gaussian pulse A(t) = sqrt(peak_power[W]) * 
            exp( -(t/T0 [ps])^2 / 2 ) centered at wavelength 
            center_wavelength (nm). time_window (ps) sets temporal grid
            size. Optional GDD and TOD are in ps^2 and ps^3."""
            
        time_window = self.internal_time_from_ps(time_window)
        T0 = self.internal_time_from_ps(T0)
        center_wavelength = self.internal_wl_from_nm(center_wavelength)
        GDD = self.internal_time_from_ps(GDD, 2)    
        TOD = self.internal_time_from_ps(TOD, 3)             
            
        # The free parameters are time_window, wavelength, and grid size.
        self.twidth = time_window          # Time window in ps
        self.center_wl = center_wavelength    # reference wavelength (nm)                        

        self.setup_grids()
        
        self.A = np.sqrt(power) * np.exp(-self.T**2/(2 * T0**2)) # input field (W^0.5)            
        self.chirp_pulse_W(GDD, TOD)
        self.chirp_pulse_T(chirp2, chirp3, T0)
        
    def gen_frog(self, time_window, center_wavelength, power, 
                 power_is_epp = False,
                 fileloc = 'O:\\OFM\\Maser\\FROG\\frog_141020-  7\\Speck.dat', # default EDFA spectrum
                 flip_phase = True):
        """Generate pulse from FROG data. Grid is centered at wavelength
        center_wavelength (nm), but pulse properties are loaded from data
        file. If flip_phase is true, all phase is multiplied by -1 [useful
        for correcting direction of time ambiguity]. time_window (ps) sets 
        temporal grid size. 
        
        power sets the pulse energy:
        if power_is_epp is True  then the number is pulse energy [J] 
        if power_is_epp is False then the power is average power [W], and 
        is multiplied by frep to calculate pulse energy"""
        try:
            self.fileloc   = fileloc
            
            time_window = self.internal_time_from_ps(time_window)            
            self.twidth    = time_window                       # Time window in ps
                  
            center_wavelength = self.internal_wl_from_nm(center_wavelength)
            self.center_wl = center_wavelength                 # reference wavelength (nm)             
            
            self.w0        = (2. * np.pi * self.c) / self.center_wl # reference angular frequency                
            
            self.setup_grids()

            if not power_is_epp:                
                power = power / self.frep
            
            # Read in retrieved FROG trace
            frog_data = np.genfromtxt(self.fileloc)                
            
            wavelengths = frog_data[:,0]# (nm)
            wavelengths = self.internal_wl_from_nm(wavelengths)
                    
            intensity   = frog_data[:,1]# (arb. units)
            phase       = frog_data[:,2]# (radians)

            if flip_phase:
                phase = -1 * phase
                
            freq_abs = self.c/wavelengths
            freq_rel = freq_abs - self.c / self.center_wl
            
            pulse_envelope = interp1d(freq_rel, intensity, kind='linear',
                                      bounds_error=False,fill_value=0)
            phase_envelope = interp1d(freq_rel, phase, kind='linear', 
                                      bounds_error=False,fill_value=0)
                                      
            gridded_intensity   = pulse_envelope(self.V/(2*np.pi))
            gridded_phase       = phase_envelope(self.V/(2*np.pi))
            
            # Calculate time domain complex electric field A
            self.A  = IFFT_t(gridded_intensity*np.exp(1j*gridded_phase))
            # Calculate normalization factor  to achieve requested 
            # pulse energy
            e_scale = np.sqrt(power / self.calc_epp() )
            self.A  = self.A * e_scale

        except IOError:
            print 'File not found.'

    def gen_noise(self, center_wl, time_window = 10.):
        
        time_window = self.internal_time_from_ps(time_window)
        center_wl = self.internal_wl_from_nm(center_wl)      
        
        self.center_wl = center_wl
        self.twidth = time_window 
        self.setup_grids()
        self.A = 1e-30 * np.ones(self.n) * np.exp(1j * 2 * np.pi * 
                 1j * np.random.rand(self.n))
                 
    def gen_CW(self, avg_power, center_wl, time_window = 10.0):
        
        time_window = self.internal_time_from_ps(time_window)
        center_wl = self.internal_wl_from_nm(center_wl)
     
        self.center_wl = center_wl
        self.twidth = time_window 
        # determine average power using energy-per-pulse mechanics        
        self.setup_grids()       
        self.A = np.ones(self.n)           
        self.A *= np.sqrt(avg_power) / sum(abs(self.get_AW()))
        
        
    def gen_OSA(self, time_window, center_wavelength, power, 
                 power_is_epp = False,
                 fileloc = 'O:\\OFM\\Maser\\Dual-Comb 100 MHz System\\Pump spectrum-Yb-101614.csv',
                 log = True, rows = 30): # Yb spectrum
                 
        """Generate pulse from OSA data. Grid is centered at wavelength
        center_wavelength (nm), but pulse properties are loaded from data
        file. time_window (ps) sets temporal grid size. Switch in place for
        importing log vs. linear data.
        
        power sets the pulse energy:
        if power_is_epp is True  then the number is pulse energy [J] 
        if power_is_epp is False then the power is average power [W], and 
        is multiplied by frep to calculate pulse energy"""
        
        try:
            self.fileloc = fileloc
            
            time_window = self.internal_time_from_ps(time_window)            
            self.twidth = time_window                       # Time window in ps
                  
            center_wavelength = self.internal_wl_from_nm(center_wavelength)
            self.center_wl = center_wavelength                 # reference wavelength (nm)             
            
            self.w0 = (2. * np.pi * self.c) / self.center_wl # reference angular frequency                
            
            self.setup_grids()
        
            if not power_is_epp:                
                power = power / self.frep
                
            # Read in OSA data
            osa_data = np.genfromtxt(self.fileloc, delimiter = ',', skiprows = rows)                
            
            wavelengths = osa_data[:,0]# (nm)
            wavelengths = self.internal_wl_from_nm(wavelengths)
                    
            intensity   = osa_data[:,1]# (arb. units)

            if log:
                intensity = 10.**(intensity / 10.)
                
            freq_abs = self.c/wavelengths
            freq_abs = np.sort(freq_abs)
            
            self.freq_rel = freq_abs - self.c / self.center_wl
            
            pulse_envelope = interp1d(self.freq_rel, intensity, kind='linear',
                                      bounds_error = False, fill_value=0)
                                      
            self.gridded_intensity = pulse_envelope(self.V / (2*np.pi))
            
            # Calculate time domain complex electric field A
            self.A  = IFFT_t(self.gridded_intensity)
            # Calculate normalization factor  to achieve requested 
            # pulse energy
            e_scale = np.sqrt(power / self.calc_epp() )
            self.A  = self.A * e_scale
                
        except IOError:
            print 'File not found.'

    def setup_grids(self):
        ''' Helper function to set up time, frequency, and wavelength grids.
            Requires:   self.twidth, self.n, self.w0
            Generates:  T, dT, dt_seconds, W, V, wl, loWL, hiWL'''
        # Calculate center angualr frequency
        self.w0 = (2.0*np.pi*self.c_nmps)/self.center_wl # reference angular frequency (2pi * time unit)
        # Create time axis                                
        self.T  = np.linspace(-self.twidth/2,self.twidth/2,self.n) # time grid
        self.dT = self.T[1] - self.T[0]        
        # Create angular frequency and wavelength axes
        self.V = 2*np.pi*np.transpose(np.arange(-self.n/2,self.n/2))/(self.n*self.dT) # Frequency grid
        self.W = self.V+self.w0 # Absolute frequency grid (THz)
        
    def calc_epp(self):
        ''' Calculate and return energy per pulse via numerical integration
            of A^2 dt'''
        return self.dT * np.trapz(abs(self.A)**2)
        
    def chirp_pulse_W(self, GDD, TOD):
        ''' Add GDD and TOD to the pulse.'''
        self.A = IFFT_t(np.exp(1j * (GDD / 2.0) * self.V**2 + 
                               1j * (TOD / 6.0) * self.V**3) * FFT_t(self.A))
                                
    def chirp_pulse_T(self, chirp2, chirp3, T0):
        self.A = self.A * np.exp(-1j * (chirp2 / 2.0) * (self.T/T0)**2 + 
                                 -1j * (chirp3 / 3.0) * (self.T/T0)**3)
    def dechirp_pulse(self, GDD_TOD_ratio = 0.0 ):

        spect_w = self.get_AW()
        phase   = np.unwrap(np.angle(spect_w))
        ampl    = np.abs(spect_w)
        mask = ampl > 0.05 * np.max(ampl)
        gdd     = np.poly1d(np.polyfit(self.W[mask], phase[mask], 2))
#        plt.figure()
#        plt.plot(self.W[mask], phase[mask])
#        plt.plot(self.W[mask], phase[mask]-gdd(self.W[mask]))
#        plt.show()
        self.A = IFFT_t(ampl * np.exp(1j*(phase-gdd(self.W))))
        
        spect_w = self.get_AW()
        phase   = np.unwrap(np.angle(spect_w))        
        print np.polyfit(self.W, phase, 2)
    def clone_pulse(self, pulse_instance):
        '''Copy all parameters of pulse_instance into this one'''
        p = pulse_instance
        self.twidth    = p.twidth
        self.center_wl = p.center_wl
        self.n         = p.n
        self.frep      = p.frep
        self.setup_grids()
        self.A         = np.copy(p.A)
    def create_cloned_pulse(self):
        '''Create and return new pulse instance identical to this instance.'''
        p = Pulse()
        p.clone_pulse(self)
        return p        
    def get_AW(self):
        return FFT_t(self.A)
        
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
