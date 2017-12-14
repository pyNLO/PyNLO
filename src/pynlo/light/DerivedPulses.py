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
import exceptions
from pynlo.light.PulseBase import Pulse

class SechPulse(Pulse):
    def __init__(self, power, T0_ps, center_wavelength_nm,
                 time_window_ps = 10., frep_MHz = 100., NPTS = 2**10, 
                 GDD = 0, TOD = 0, chirp2 = 0, chirp3 = 0,
                 power_is_avg = False):
        """Generate sech pulse A(t) = sqrt(P0 [W]) * sech(t/T0 [ps])
        centered at wavelength center_wavelength_nm (nm).
        time_window (ps) sets temporal grid size. Optional GDD and TOD are
        in ps^2 and ps^3."""
        Pulse.__init__(self, frep_MHz = frep_MHz, n = NPTS)
        # make sure we weren't passed mks units        
        assert (center_wavelength_nm > 1.0) 
        assert (time_window_ps > 1.0 )                
        self.set_center_wavelength_nm(center_wavelength_nm)        
        self.set_time_window_ps(time_window_ps)
                        
        ### Generate pulse
        if not power_is_avg:
            self.set_AT( np.sqrt(power)/np.cosh(self.T_ps/T0_ps) )
        else:
            self.set_AT( 1 / np.cosh(self.T_ps/T0_ps) )
            self.set_AT(self.AT * np.sqrt( power / ( frep_MHz*1.0e6 * self.calc_epp()) ))
        
        self.chirp_pulse_W(GDD, TOD)
        self.chirp_pulse_T(chirp2, chirp3, T0_ps)            
        
class GaussianPulse(Pulse):
    def __init__(self, power, T0_ps, center_wavelength_nm,
                 time_window_ps = 10., frep_MHz = 100., NPTS = 2**10, 
                 GDD = 0, TOD = 0, chirp2 = 0, chirp3 = 0,
                 power_is_avg = False):
        """Generate Gaussian pulse A(t) = sqrt(peak_power[W]) * 
            exp( -(t/T0 [ps])^2 / 2 ) centered at wavelength 
            center_wavelength_nm (nm). time_window (ps) sets temporal grid
            size. Optional GDD and TOD are in ps^2 and ps^3."""

        Pulse.__init__(self, frep_MHz = frep_MHz, n = NPTS)
        # make sure we weren't passed mks units        
        assert (center_wavelength_nm > 1.0) 
        assert (time_window_ps > 1.0 )        
        self.set_center_wavelength_nm(center_wavelength_nm)
        self.set_time_window_ps(time_window_ps)        
        
        GDD = GDD
        TOD = TOD
                   
        self.set_AT( np.sqrt(power) * np.exp(-2.77*self.T_ps**2/(T0_ps**2)) ) # input field (W^0.5) 
        if power_is_avg:            
            self.set_AT(self.AT * np.sqrt( power / ( frep_MHz*1.0e6 * self.calc_epp()) ))
        self.chirp_pulse_W(GDD, TOD)
        self.chirp_pulse_T(chirp2, chirp3, T0_ps)
    
class SincPulse(Pulse):
    def __init__(self, power, FWHM_ps, center_wavelength_nm,
                 time_window_ps = 10., frep_MHz = 100., NPTS = 2**10, 
                 GDD = 0, TOD = 0, chirp2 = 0, chirp3 = 0,
                 power_is_avg = False):
        """Generate sinc pulse A(t) = sqrt(peak_power[W]) * sin(t/T0)/(t/T0)
        centered at wavelength center_wavelength_nm (nm).
        The width is given by FWHM_ps, which is the full-width-at-half-maximum 
        in picoseconds. T0 is equal th FWHM/3.7909885.
        time_window_ps sets temporal grid size. Optional GDD and TOD are
        in ps^2 and ps^3."""
        Pulse.__init__(self, frep_MHz = frep_MHz, n = NPTS)
        # make sure we weren't passed mks units        
        assert (center_wavelength_nm > 1.0) 
        assert (time_window_ps > 1.0 )                
        self.set_center_wavelength_nm(center_wavelength_nm)        
        self.set_time_window_ps(time_window_ps)

        T0_ps = FWHM_ps/3.7909885
        ### Generate pulse
        if not power_is_avg:
            # numpy.sinc is sin(pi*x)/(pi*x), so we divide by pi
            self.set_AT( np.sqrt(power) * np.sinc(self.T_ps/(T0_ps*np.pi)) ) 
        else:
            self.set_AT( 1 / np.sinc(np.pi * self.T_ps/(T0_ps*np.pi)) )
            self.set_AT(self.AT * np.sqrt( power / ( frep_MHz*1.0e6 * self.calc_epp()) ))
        
        self.chirp_pulse_W(GDD, TOD)
        self.chirp_pulse_T(chirp2, chirp3, T0_ps)   
        
class FROGPulse(Pulse):
    def __init__(self, time_window_ps, center_wavelength_nm, power,frep_MHz = 100., NPTS = 2**10,
                 power_is_avg = False,
                 fileloc = '',
                 flip_phase = True):
        """Generate pulse from FROG data. Grid is centered at wavelength
        center_wavelength_nm (nm), but pulse properties are loaded from data
        file. If flip_phase is true, all phase is multiplied by -1 [useful
        for correcting direction of time ambiguity]. time_window (ps) sets 
        temporal grid size. 
        
        power sets the pulse energy:
        if power_is_epp is True  then the number is pulse energy [J] 
        if power_is_epp is False then the power is average power [W], and 
        is multiplied by frep to calculate pulse energy"""
        Pulse.__init__(self, frep_MHz = frep_MHz, n = NPTS)
        try:
            self.fileloc   = fileloc
            # make sure we weren't passed mks units
            assert (center_wavelength_nm > 1.0) 
            assert (time_window_ps > 1.0 )
            self.set_time_window_ps(time_window_ps)
            self.set_center_wavelength_nm(center_wavelength_nm) # reference wavelength (nm)                         
            
            # power -> EPP
            if power_is_avg:
                power = power / self.frep_mks
            
            # Read in retrieved FROG trace
            frog_data = np.genfromtxt(self.fileloc)
            wavelengths = frog_data[:,0]# (nm)
            intensity   = frog_data[:,1]# (arb. units)
            phase       = frog_data[:,2]# (radians)

            if flip_phase:
                phase = -1 * phase
                            
            pulse_envelope = interp1d(wavelengths, intensity, kind='linear',
                                      bounds_error=False,fill_value=0)
            phase_envelope = interp1d(wavelengths, phase, kind='linear', 
                                      bounds_error=False,fill_value=0)
                                      
            gridded_intensity   = pulse_envelope(self.wl_nm)
            gridded_phase       = phase_envelope(self.wl_nm)

            # Calculate time domain complex electric field A
            self.set_AW(gridded_intensity*np.exp(1j*gridded_phase))
            # Calculate normalization factor  to achieve requested 
            # pulse energy
            e_scale = np.sqrt(power / self.calc_epp() )
            self.set_AT(self.AT * e_scale )

        except IOError:
            print 'File not found.'

class NoisePulse(Pulse):
    def __init__(self, center_wavelength_nm, time_window_ps = 10., NPTS = 2**8,
                 frep_MHz = None):
        Pulse.__init__(self, n = NPTS, frep_MHz = frep_MHz)
        self.set_center_wavelength_nm(center_wavelength_nm)
        self.set_time_window_ps(time_window_ps)        
        
        self.set_AW( 1e-30 * np.ones(self.NPTS) * np.exp(1j * 2 * np.pi *
                 1j * np.random.rand(self.NPTS)))
                 
class CWPulse(Pulse):
    def __init__(self, avg_power, center_wavelength_nm, time_window_ps = 10.0,
                 NPTS = 2**8,offset_from_center_THz = None):
        Pulse.__init__(self, n = NPTS)
        # make sure we weren't passed mks units
        assert (center_wavelength_nm > 1.0) 
        assert (time_window_ps > 1.0 )        

        if offset_from_center_THz is None:            
            self.set_center_wavelength_nm(center_wavelength_nm)
            self.set_time_window_ps(time_window_ps)        
         
            # Set the time domain to be CW, which should give us a delta function in
            # frequency. Then normalize that delta function (in frequency space) to
            # the average power. Note that frep does not factor in here.
            self.set_AT(np.ones(self.NPTS,))
            self.set_AW(self.AW * np.sqrt(avg_power) / sum(abs(self.AW)) )
        else:
            dF = 1.0/time_window_ps
            n_offset = np.round( offset_from_center_THz/dF)      
            
            center_THz = self._c_nmps/center_wavelength_nm -\
                                    n_offset * dF
            center_nm = self._c_nmps / center_THz
            self.set_time_window_ps(time_window_ps)
        
            self.set_center_wavelength_nm(center_nm)
            aws = np.zeros((self.NPTS, ))
            aws[int(self.NPTS/2.0) + int(n_offset)  ] = 1.0 *np.sqrt(avg_power)
            self.set_AW(aws)
        
    def gen_OSA(self, time_window_ps, center_wavelength_nm, power, 
                 power_is_epp = False,
                 fileloc = 'O:\\OFM\\Maser\\Dual-Comb 100 MHz System\\Pump spectrum-Yb-101614.csv',
                 log = True, rows = 30): # Yb spectrum
                 
        """Generate pulse from OSA data. Grid is centered at wavelength
        center_wavelength_nm (nm), but pulse properties are loaded from data
        file. time_window (ps) sets temporal grid size. Switch in place for
        importing log vs. linear data.
        
        power sets the pulse energy:
        if power_is_epp is True  then the number is pulse energy [J] 
        if power_is_epp is False then the power is average power [W], and 
        is multiplied by frep to calculate pulse energy"""
        
        try:
            self.fileloc = fileloc
            
            self.set_time_window_ps(time_window_ps)
                  
            self.center_wl = center_wavelength_nm                 # reference wavelength (nm)             
            
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
