# -*- coding: utf-8 -*-
"""
Created on Wed Jun 03 15:11:20 2015
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
import exceptions
import numpy as np
from scipy import misc, optimize
from scipy.constants import speed_of_light

class Crystal:
    """ Container for chi-2 nonlinear crystals. Actual crystal refractive index,
    dispersion, and nonlinearity information is stored in modular files. Read these
    in by calling <crystal>.load(crystal_instance, params). """
    _c_nm_ps     = speed_of_light*1e9/1e12 # c in nm/ps
    # Cache results
    _wavelength_axes    = {}
    _enable_caching     = False
    _cached_ns          = {}
    _crystal_properties  = {'damage_threshold_GW_per_sqcm': 1.0,
                            'damage_threshold_info' : ''}
    
    def __init__(self, params):                
        if params.has_key('length'):            
            self._length = params['length']
        else:
            self._length = 1.0
        if params.has_key('enable_caching'):
            self._enable_caching = params['enable_caching']
        else:
            self._enable_caching = False
            
    def set_pp_chirp(self, start, stop):
        self.pp   = lambda(x): start + (stop-start) * x /self.length        
    def get_pulse_k(self, pulse_instance, axis = None):
        """ Return vector of angular wavenumbers (m^-1) for the pulse_instance's 
            frequency grid inside the crystal """
        if axis is None:
            ks = 2.0 * np.pi * self.n(pulse_instance.wl_nm) / pulse_instance.wl_mks
        else:
            ks = 2.0 * np.pi * self.n(pulse_instance.wl_nm, axis) / pulse_instance.wl_mks
        return ks            
    def get_pulse_n(self, pulse_instance, axis = None):
        """ Return vector of indices of refraction for the pulse_instance's 
            frequency grid inside the crystal """
        if self._enable_caching:
            if self._cached_ns.has_key(pulse_instance.cache_hash + str(axis)):
                return self._cached_ns[pulse_instance.cache_hash + str(axis)]
            else:
                if axis is None:
                    ns = self.n(pulse_instance.wl_nm)
                else:
                    ns = self.n(pulse_instance.wl_nm, axis)
                self._cached_ns[pulse_instance.cache_hash + str(axis)]= ns
        else:
            if axis is None:
                ns = self.n(pulse_instance.wl_nm)
            else:
                ns = self.n(pulse_instance.wl_nm, axis)
        return ns
    def calculate_group_velocity_nm_ps(self, wavelengths_nm, axis = None):
        """ Calculate group velocity vg at 'wavelengths_nm' [nm] along 'axis'
            in units of nm/ps """
        # Equation 4.7.7b in Verdeyen
        fn = lambda(x): self.n(x, axis)
        dn_dl = misc.derivative(fn, wavelengths_nm, dx = 0.1, n = 1, order = 11)
        vg_inverse = (1.0 / self._c_nm_ps) * (fn(wavelengths_nm) - wavelengths_nm * dn_dl)
        return 1.0 / vg_inverse
    def calculate_pulse_delay_ps(self, wl1_nm, wl2_nm, crystal_length_mks = None, axis = None):
        """ Calculate the pulse delay between pulses at wl1 and wl2 after
            crystal. Be default, crystal instance's length is used. """
        if crystal_length_mks is None:
            crystal_length = self.length_nm
        else:
            crystal_length = 1.0e9 * crystal_length_mks
        vg1 = self.calculate_group_velocity_nm_ps(wl1_nm, axis)
        vg2 = self.calculate_group_velocity_nm_ps(wl2_nm, axis)
        delta_t = crystal_length/vg1 - crystal_length/vg2
        return delta_t        
        
    def calculate_D_ps_nm_km(self, wavelengths_nm, axis = None):
        """ Calculate crystal dispersion at 'wavelengths_nm' [nm] along 'axis' in
            standard photonic engineering units ps/nm/km"""        
        fn = lambda(x): self.n(x, axis)
        d2n_dl2 = misc.derivative(fn, wavelengths_nm, dx = 0.1, n = 2, order = 11)
        D1      = (wavelengths_nm / self._c_nm_ps) * d2n_dl2 # units are ps/nm/nm
        D       = D1 * 1.0e12
        return D    
    def calculate_D_fs_um_mm(self, wavelengths_nm, axis = None):
        """ Calculate crystal dispersion at 'wavelengths_nm' along 'axis' in
            short crystal, broad bandwidth units of fs/um/mm """
        D = self.calculate_D_ps_nm_km(wavelengths_nm, axis)
        scale = 1.0
        return D * scale
    def calculate_mix_phasematching_bw(self, pump_wl_nm, signal_wl_nm, axis = None ):
        r"""Calculate the phase matching bandwidth in the case of mixing
            between narrowband pump (highest photon energy) with a signal field.
            The bandwidths of mixing between pump-signal and pump-idler are
            calculated, and the smaller of the two is returned.
        
        Parameters
        ----------
        pump_wl_nm : float
             Wavelength of pump field, bandwidth assumed to be 0 [nm]
        signal_wl_nm : array-like
             Wavelength of signal field [nm]

        Returns
        -------
        acceptance bandwidth : float
            Phasematching bandwidth [m^-1 * m]
            
        References
        ----------
        Peter E Powers, "Fundamentals of Nonlinear Optics", pp 106
        """
        idler_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/signal_wl_nm)
        vg_s = self.calculate_group_velocity_nm_ps(signal_wl_nm, axis)*1e3 # m/s
        vg_i = self.calculate_group_velocity_nm_ps(idler_wl_nm,  axis)*1e3 # m/s
        vg_p = self.calculate_group_velocity_nm_ps(pump_wl_nm,   axis)*1e3 # m/s
        try:
            vg_p = np.ones((len(vg_s),)) * vg_p
        except TypeError:
            pass
        
        deltaOmega_deltaL = np.minimum(np.abs( 0.886 * np.pi / (1.0/vg_s - 1.0/vg_p) ),
                                   np.abs( 0.886 * np.pi / (1.0/vg_i - 1.0/vg_p) ) )
        
        
        deltak_deltaL = deltaOmega_deltaL * 1.0/speed_of_light
        return deltak_deltaL

    def invert_dfg_qpm_to_signal_wl(self, pump_wl_nm, poling_period_mks, 
                                    max_signal_wl_nm = 2000 ):
        r"""Calculate the signal wavelength phasematched in QPM by the given
            poing period for the specified pump wavelength.
        
        Parameters
        ----------
        pump_wl_nm : float
             Wavelength of pump field, bandwidth assumed to be 0 [nm]
        poling_period_mks : float
             Period length of the QPM grating

        Returns
        -------
        Signal wavelength [nm] : float
            
        """

        def err_fn(wl_s):
            return (self.calculate_poling_period(pump_wl_nm, wl_s, None, silent = True)[0] - poling_period_mks )**2
        
        res = optimize.minimize_scalar(err_fn, bounds = [pump_wl_nm*1.001, max_signal_wl_nm],
                                 method = 'bounded')
        
        return res.x

    def set_caching(self, cache_enable = True):
        r""" Enable or disable caching of refractive indices. Enabling this uses
        more memory, but can save costly recomputations
        
        Parameters
        ----------
        cache_enable : bool
        """
        assert(cache_enable == True or cache_enable == False)
        self._enable_caching = cache_enable
    def _get_length_mks(self):
        return self._length
    def _get_length_nm(self):
        return self._length * 1.0e9
    length_mks = property(_get_length_mks)    
    length_nm = property(_get_length_nm)    
    def _damage_threshold_mks(self):
        return self._crystal_properties['damage_threshold_GW_per_sqcm'] * 1.0e13
    damage_threshold_mks = property(_damage_threshold_mks)