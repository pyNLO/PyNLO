# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:08:31 2015
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
from scipy import constants

class OneDBeam_highV_WG:
    """ Class for propagation and calculating field intensities in a waveguide.
        Contains beam shape and propagation axis information. The mode area is
        held constant for all colors, and does not change with z.
        """
    _Aeff = 1.0
    _lambda0 = None
    _crystal_ID = None
    _n_s_cache  = None
    
    def __init__(self, Aeff_squm = 10.0, this_pulse = None, axis = None):
        """ Initialize class instance. Calculations are done from the effective
            area. """
        self._lambda0 = this_pulse.wl_mks
        self.axis   = axis
        self.set_Aeff( Aeff_squm*1e-12 )

      
    def set_Aeff(self, Aeff):
        self._Aeff = Aeff
        
    def _get_Aeff(self):
        return self._Aeff
        
    Aeff  = property(_get_Aeff)
    
        
    def calculate_gouy_phase(self, z, n_s):
        """ Return the Gouy phase shift, which in a waveguide is constant (1.0)"""
        return 1.0
        
    def _rtP_to_a(self, n_s, z, waist = None):
        """ Calculate conversion constant from electric field to average power from
            indices of refraction: A = P_to_a * rtP """
        return 1.0 / np.sqrt( self._Aeff * n_s * \
                        constants.epsilon_0 * constants.speed_of_light)
                         
    def rtP_to_a(self, n_s, z = None):
        """ Calculate conversion constant from electric field to average power from
            pulse and crystal class instances: A ** 2 = rtP_to_a**2 * P """
        return self._rtP_to_a(n_s, z)
        
    def rtP_to_a_2(self, pulse_instance, crystal_instance, z = None, waist = None):
        """ Calculate conversion constant from electric field to average power from
            pulse and crystal class instances: A ** 2 = rtP_to_a**2 * P """
        n_s = self.get_n_in_crystal(pulse_instance, crystal_instance)
        return self._rtP_to_a(n_s, z)
        
    def calc_overlap_integral(self, z, this_pulse, othr_pulse, othr_beam,\
                                   crystal_instance, reverse_order = False):
        """ Calculate overlap integral (field-square) between this beam and  Beam instance
            second_beam inside of a crystal. In a high V number waveguide, the 
            modes have the same size, so 1.0 is returned."""
        return  1.0
        

    def get_n_in_crystal(self, pulse_instance, crystal_instance):
        return crystal_instance.get_pulse_n(pulse_instance, self.axis)
        
    def get_k_in_crystal(self, pulse_instance, crystal_instance):
        return crystal_instance.get_pulse_k(pulse_instance, self.axis)