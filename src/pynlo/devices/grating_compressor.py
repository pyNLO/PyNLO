# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 09:40:41 2015
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
from scipy import constants, misc, signal, integrate
import exceptions

class TreacyCompressor:
    """ This class calculates the effects of a grating-based pulse compressor,
        as described in 
        E. B. Treacy, "Optical Pulse Compression With Diffraction Gratings",
        IEEE Journal of Quantum Electronics QE5(9), p454 (1969):
        http://dx.doi.org/10.1109/JQE.1969.1076303
        
        It implements eqn 5b from Treacy1969: ::
        
                                     -4 pi^2 c b 
        {1}  dt/dw =    -------------------------------------
                       w^3 d^2 (1- (2 pi c/ wd - sin gamma)^2)
                       

        where gamma is the diffraction angle, w is the angular frequency, d is
        the grating ruling period, and b is the slant distance between gratings, ::
        
        {1b} b = G sec(gamma - theta)
        
        where G is the grating separation and theta is the acute angle between
        indicent and diffracted rays (text before eq 4). The grating equation ::
        relates the angles (generalized eq 3):
        
        {2} sin(gamma - theta) + sin(gamma) = m lambda / d

        More conventionally, the grating equation is cast in terms of the
        incident and diffracted ray angles, ::
        
        {3} sin(alpha) + sin(beta) = m lambda / d.

        It makes sense to solve {3} using the grating specifications (eg for
        optimum incident angle a) and then derive Treacy's theta and gamma: ::
        
        {4} gamma = alpha                  theta = gamma - alpha
        
        This code only considers first order diffraction, as most gratings are
        designed for this (eg LightSmyth transmission gratings.)
        """
    d = 0.0 # grating period (meters)
    g = 0.0 # incident beam angle wrt grating
    
    def __init__(self, lines_per_mm, incident_angle_degrees):
        """ Initialize with the two parameters intrinsic to the grating, the
            ruling density and design angle of incidence. """
        self.d = 1.0e-3 / lines_per_mm
        self.g = incident_angle_degrees * 2.0*np.pi / 360.0
        
    def calc_theta(self, wavelength_nm, display_angle = False):
        l = wavelength_nm * 1.0e-9
        # First solve the grating equation {3} for the diffracted angle
        if np.any((l/self.d  - np.sin(self.g))< -1) or np.any((l/self.d  - np.sin(self.g)) > 1):
            print "Bad value for argument of arcsin: ",\
                l/self.d  - np.sin(self.g),'. You are probably asking for diffraction of an impossible color (this wavelength is ',l*1e9,'nm. Coercing to [-1,1].'
        val = l/self.d  - np.sin(self.g)
        if type(val) == np.ndarray:
            val[val>1] = 1
            val[val<-1] = -1

        alpha = np.arcsin(val )
        if display_angle:
            print 'diffraction angle = ',alpha * 360.0/(2.0*np.pi)
        # Then calculate gamma (ok, this is not much work because b = gamma).
        # Calculate theta from {4}:
        theta = self.g-alpha
        return theta
    def calc_dt_dw_singlepass(self, wavelength_nm,
                              grating_separation_meters,
                              verbose = False):
        c = constants.speed_of_light
        G = grating_separation_meters
        l = wavelength_nm * 1.0e-9         
        w = 2.0 * np.pi * c / l        
        theta = self.calc_theta(wavelength_nm, display_angle = verbose)
        gamma = self.g
        b = G / np.cos(gamma - theta)
        
        return (-4.0 * np.pi**2 * c * b)/ (w**3 * self.d**2 *
                (1.0 - (2.0*np.pi*c/(w*self.d) - np.sin(gamma))**2 ))
                
    def calc_dphi_domega(self, omega,
                              grating_separation_meters,
                              verbose = False):        
        c = constants.speed_of_light
        wavelength_nm = 1.0e9*2.0 * np.pi * c / omega
        G = grating_separation_meters
        theta = self.calc_theta(wavelength_nm, display_angle = verbose)
        gamma = self.g
        b = G / np.cos(gamma - theta)
        p = b*(1.+np.cos(theta))
        return p/c
        
    def calc_compressor_gdd(self, wavelength_nm, grating_separation_meters):
        return 2.0 * self.calc_dt_dw_singlepass(wavelength_nm,
                                                 grating_separation_meters,
                                                 verbose = False)
    
    
    def calc_compressor_HOD(self, wavelength_nm, grating_separation_meters, dispersion_order):
        """ Calculate higher order dispersion by taking w - derivatives of
            dt/dw """        
        if dispersion_order < 3:
            raise exceptions.ValueError('Order must be > 2. For TOD, specify 3.')
        w_of_l = lambda x: 2.0*np.pi*constants.speed_of_light / (x*1.0e-9)
        l_of_w = lambda x: 1.0e9*2.0*np.pi*constants.speed_of_light / x
        
        fn = lambda x:self.calc_compressor_gdd(l_of_w(x), grating_separation_meters)        
        return misc.derivative(fn, 
                                      w_of_l(wavelength_nm),
                                      n     = dispersion_order - 2,
                                      dx    = 2.0*np.pi*100.0e6, # Use dx = 100 MHz
                                      order = 101 ) # Why not use 101's order?
    
    
    def calc_compressor_dnphi_domega_n(self, wavelength_nm, grating_separation_meters, dispersion_order):
        """ Calculate higher order dispersion by taking w - derivatives of
            dt/dw """        
        if dispersion_order < 1:
            raise exceptions.ValueError('Order must be > 2. For GDD, specify 1.')
        w_of_l = lambda x: 2.0*np.pi*constants.speed_of_light / (x*1.0e-9)    
        fn = lambda x:self.calc_dphi_domega(x, grating_separation_meters)        
        w0 = w_of_l(wavelength_nm)        
        ws = np.linspace(w0 - 10.0e12,w0 + 10.0e12, 101) 
        dphidw = fn(ws)        
        y = signal.savgol_filter(dphidw, window_length = 11,
                             polyorder = 7, deriv = dispersion_order)
        dw = (ws[1] - ws[0])*10**(-15*(1+dispersion_order))
        return y[50]/(dw**dispersion_order)
    
    
    def apply_phase_to_pulse(self, grating_separation_meters, pulse):
        """ Apply grating disersion (all orders) to a Pulse instance. Phase is
            computed by numerical integration of dphi/domega (from Treacy) """
        w0 = pulse.center_frequency_THz * 2.0*np.pi*1.0e12
        integrand = lambda x: 2.0 * self.calc_dphi_domega(x, grating_separation_meters)
        calc_phase = lambda x:integrate.quad(integrand, w0, x, 
                                             epsabs = 1.0e-8,epsrel = 1.0e-8,)[0]
        vec_calc_phase = np.vectorize(calc_phase)
        phase = vec_calc_phase(pulse.W_mks)
        groupdelay = np.polyder(np.polyfit(pulse.W_mks, phase, 2))[0]
        pulse.apply_phase_W(phase + pulse.V_mks * groupdelay)
        