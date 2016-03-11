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
from scipy import constants, optimize

class OneDBeam:
    """ Simple Gaussian beam class for propagation and calculating field 
        intensities. Contains beam shape and propagation axis information. The
        beam waist is held independent of index of refraction, from which the
        confocal parameter and beam geometry can be calculated.
        
        According to Boyd, who cites Klienman (1966) and Ward and New (1969),
        it is generally true that the confocal parameter is conserved in
        harmonic generation and DFG. This parameter is 
        b = 2 pi w0**2 / lambda."""
    _w0 = 1.0
    _lambda0 = None
    _crystal_ID = None
    _n_s_cache  = None
    
    def __init__(self, waist_meters = 1.0, this_pulse = None, axis = None):
        """ Initialize class instance. From waist, confocal parameter is derived.
            A Pulse class is input, and it is assumed that each color focuses to
            the same waist size at the same point. From this, the (chromatic) confocal 
            parameter b(lambda) is calculated"""
        self._lambda0 = this_pulse.wl_mks
        self.axis   = axis
        self.set_w0( waist_meters )

    def calc_confocal(self, n_s = 1.0):
        return (2.0*np.pi) * self.waist**2 * (n_s/self._lambda0)  
      
    def set_w0(self, w0):
        self._w0 = w0
        
    def _get_w0(self):
        return self._w0
        
    waist  = property(_get_w0)
    
    def calculate_waist(self, z, n_s = 1.0):
        """ Calculate the beam waist a distance z from the focus. The expression
            is :
                w(z) = w0 (1+ ( 2z/b)**2 )**1/2 """
        b = self.calc_confocal(n_s)
        return self.waist * np.sqrt(1. + ( 2.0* z / b)**2 )
        
    def calculate_zR(self, n_s = 1.0):
        """ Calculate Rayleigh range, accounting for index of refraction. """
        return self.calc_confocal(n_s) / 2.0
        
    def calculate_R(self, z, n_s = 1.0):
        """ Calculate beam curvature. :
            R(z) = z * [ 1 +  (z_R/ z)**2 ]"""
        z_r = self.calculate_zR(n_s)
        return z * (1 + (z_r/z)**2)
        
    def calculate_gouy_phase(self, z, n_s):
        """ Return the Gouy phase shift due to focusing a distance z in a crystal,
            where it is assumed that the focus is at crystal_length / 2.0. Return
            is exp(i psi), as in eq 37 in Siegman Ch 17.4, where A ~ exp(-ikz + i psi)."""
        z_r      = self.calculate_zR(n_s)
        psi_gouy = np.arctan2(z, z_r )
        return np.exp(1j*psi_gouy)
        
    def _rtP_to_a(self, n_s, z, waist = None):
        """ Calculate conversion constant from electric field to average power from
            indices of refraction: A = P_to_a * rtP """
        if waist is None:
            waist = self.calculate_waist(z, n_s)
        return 1.0 / np.sqrt( np.pi * waist**2 * n_s * \
                        constants.epsilon_0 * constants.speed_of_light)   
                         
    def rtP_to_a(self, n_s, z = None):
        """ Calculate conversion constant from electric field to average power from
            pulse and crystal class instances: A ** 2 = rtP_to_a**2 * P """
        return self._rtP_to_a(n_s, z, self.waist)
        
    def rtP_to_a_2(self, pulse_instance, crystal_instance, z = None, waist = None):
        """ Calculate conversion constant from electric field to average power from
            pulse and crystal class instances: A ** 2 = rtP_to_a**2 * P """
        n_s = self.get_n_in_crystal(pulse_instance, crystal_instance)
        return self._rtP_to_a(n_s, z, waist)
        
    def calc_overlap_integral(self, z, this_pulse, othr_pulse, othr_beam,\
                                   crystal_instance, reverse_order = False):
        """ Calculate overlap integral (field-square) between this beam and  Beam instance
            second_beam inside of a crystal. If reverse_order is true, then the 
            order of second_beam will be reversed. """
        n1 = self.get_n_in_crystal(this_pulse, crystal_instance)
        n2 = othr_beam.get_n_in_crystal(othr_pulse, crystal_instance)
        zr1 = self.calculate_zR(n1)
        zr2 = othr_beam.calculate_zR(n2)
        k1 = self.get_k_in_crystal(this_pulse, crystal_instance)
        k2 = othr_beam.get_k_in_crystal(othr_pulse, crystal_instance)

        # This expression only accounts for beam size
        #return (2*w2*w2/(w1**2+w2**2))**2
        # Expression below accounts for curvature:       
        return  (4*k1*k2*zr1*zr2)/(k2**2*(z**2 + zr1**2) - 2*k1*k2*(z**2 - zr1*zr2) + k1**2*(z**2 + zr2**2))
        
    def set_waist_to_match_confocal(self, this_pulse, othr_pulse, othr_beam,\
                                   crystal_instance):
        """ Calculate waist w0 for a beam match confocal parameters with othr_beam """

        n1 = self.get_n_in_crystal(this_pulse, crystal_instance)
        n2 = othr_beam.get_n_in_crystal(othr_pulse, crystal_instance)
        zr = othr_beam.calculate_zR(n2)  
        w0 = np.sqrt( 2.0*zr*self._lambda0/(2.0*np.pi*n1))
        self.waist = w0
        
    def set_waist_to_match_central_waist(self, this_pulse,w0_center,crystal_instance):
        """ Calculate waist w0 for a beam match so that all confocal parameters
            are equal while matching waist w0_center at center color of this beam  """

        n1 = self.get_n_in_crystal(this_pulse, crystal_instance)
        zr = (np.pi) * w0_center**2 * (n1[len(n1)>>1]/self._lambda0[len(self._lambda0)>>1])
        w0 = np.sqrt( 2*zr*self._lambda0/(2.0*np.pi*n1))
        self.waist = w0
        
    def calc_optimal_beam_overlap_in_crystal(self, this_pulse, othr_pulse, othr_beam,\
                                   crystal_instance, L = None):
        """ Calculate waist w0 for a beam to maximuze the integral (field-square) 
            between it beam and  Beam instance second_beam integrated along the
            length of a crystal. If L is not specified, then the crystal length
            is used. """
        if L is None:
            L = crystal_instance.length_mks
        n1 = self.get_n_in_crystal(this_pulse, crystal_instance)
        n2 = othr_beam.get_n_in_crystal(othr_pulse, crystal_instance)
        
        zr2 = othr_beam.calculate_zR(n2)
        k1 = self.get_k_in_crystal(this_pulse, crystal_instance)
        k2 = othr_beam.get_k_in_crystal(othr_pulse, crystal_instance)

        obj_fn = lambda(zr1): -1.0*np.sum((4*k1*k2*zr1*abs(zr2) *\
            np.arctan( ((k1 - k2)*L)/(k2*zr1 + k1*abs(zr2)))/\
            ((k1 - k2)*(k2*zr1 + k1*abs(zr2)))))
        
        result = optimize.minimize(obj_fn, zr2,  method = 'Powell')
        # From w0**2 = b lambda/ 2 pi n:
        w0_out = np.sqrt( 2.0 *result.x*self._lambda0/(2.0*np.pi*n1))
        return w0_out

    def get_n_in_crystal(self, pulse_instance, crystal_instance):
        return crystal_instance.get_pulse_n(pulse_instance, self.axis)
        
    def get_k_in_crystal(self, pulse_instance, crystal_instance):
        return crystal_instance.get_pulse_k(pulse_instance, self.axis)