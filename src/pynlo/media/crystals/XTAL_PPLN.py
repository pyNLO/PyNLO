# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:23:17 2015

Sellemeier coefficients and nonlinear parameter for PPLN
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
from pynlo.media.crystals.CrystalContainer import Crystal

class DengSellmeier:
    """ Temperature dependent refractive index for e axis of PPLN, using 
    equations from Deng et al."""
    a1  = 5.39121
    a2  = 0.100473
    a3  = 0.20692
    a4  =  100.0
    a5  = 11.34927
    a6  =  1.544e-2
    b1  = 	 4.96827e-7
    b2  = 	 3.862e-8
    b3  = 	 -0.89e-8
    b4  = 	 2.657e-5
    b5  =   9.62119e-10
    T   = 0
    def __init__(self, T):
        self.set_T_degC(T)
    def set_T_degC(self, T):
        self.T = T
    def n(self, wl_nm, axis = None):
        wl_um = wl_nm * 1.0e-3
        f = (self.T - 24.5)*(self.T+570.82)
        return np.sqrt(self.a1 + self.b1*f +\
                (self.a2 + self.b2*f)/(wl_um**2-(self.a3+self.b3*f)**2) +\
                (self.a4 + self.b4*f)/(wl_um**2 - self.a5**2) -\
                (self.a6 + self.b5*f) * wl_um**2)
class Gayer5PctSellmeier:
    """ Temperature dependent refractive index for e axis of PPLN, 5pct Mg, 
        using equations from Gayer et al."""    
    a1  = 5.756
    a2  = 0.0983
    a3  = 0.2020
    a4  =  189.32
    a5  = 12.52
    a6  =  1.32e-2
    b1  = 	 2.860e-6
    b2  = 	 4.700e-8
    b3  = 	 6.113e-8
    b4  = 	 1.526e-5
    T   =    30
    def __init__(self, T):
        self.set_T_degC(T)    
    def set_T_degC(self, T):
        self.T = T        
    def n(self, wl_nm, axis = None):
        wl_um = wl_nm * 1.0e-3
        f = (self.T - 24.5)*(self.T+570.82)
        return np.sqrt(self.a1 + self.b1*f +\
                (self.a2 + self.b2*f)/(wl_um**2-(self.a3+self.b3*f)**2) +\
                (self.a4 + self.b4*f)/(wl_um**2 - self.a5**2) -\
                self.a6 * wl_um**2)        
class PPLN(Crystal):
    
    def __init__(self, T, **params):
        Crystal.__init__(self, params)
        self.load(T)
    def load(self, T, data_source = "Gayer_5pctMg"):
        """ Load PPLN data. params -- 'T' : crystal temperature
        Uses parameters from:
        * Deng: Deng et al, Opt. Comm. 268, 1, 1 pp 110-114
            'Improvement to Sellmeier equation for periodically poled LiNbO3
            crystal using mid-infrared difference-frequency generation'
        * Gayer_5pctMg: Appl. Phys. B 91, 343–348 (2008) 
            'Temperature and wavelength dependent refractive index equations 
            for MgO-doped congruent and stoichiometric LiNbO3'
        """
        self.T = T
        self.mode = 'PP'
        self.sellmeier_type = data_source
        
        
        self.sellmeier_calculators = {'Deng' :DengSellmeier(T),
                                      'Gayer_5pctMg':Gayer5PctSellmeier(T)}
        self.n = self.sellmeier_calculators[data_source].n
        self.set_xtalT = self.sellmeier_calculators[data_source].set_T_degC
        
        self.deff=  14.9e-12 # from SNLO
        self.n2=   3e-15 / 100**2 # from Nikogosyan
        self.pp=   lambda(x): 30.49e-6 
        self._crystal_properties['damage_threshold_GW_per_sqcm'] = 4.0
        self._crystal_properties['damage_threshold_info'] = """ This 4 GW/cm^2 number is from Covesion. According
        to their website, it is from a 200 fs pulses source at 1550 nm."""
    def set_pp(self, p) :
        if p.__class__ is tuple:
            self.pp = lambda(x): p[0]
        else:
            self.pp = lambda(x): p(x)
    def set_T(self, T_degC):
        self.T = T_degC
        self.set_xtalT(T_degC)
        
    def calculate_poling_period(self, pump_wl_nm, sgnl_wl_nm, idlr_wl_nm, 
                                delta_k_L  = 3.2, silent=False):
        """ Calculate poling period [meters] for pump, signal, and idler -- each a 
            PINT object (with units.) If one is None, then it is calculated by
            energy conservation. """
        RET_wl_nm = False
        new_wl_nm = None
        if pump_wl_nm is None:
            pump_wl_nm = 1.0/(1.0/idlr_wl_nm + 1.0/sgnl_wl_nm)
            if not silent:
                print 'Setting pump to ',pump_wl_nm
            RET_wl_nm = True
            new_wl_nm = pump_wl_nm
        if sgnl_wl_nm is None:
            sgnl_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/idlr_wl_nm)
            if not silent:
                print 'Setting signal to ',sgnl_wl_nm
            RET_wl_nm = True
            new_wl_nm = sgnl_wl_nm
        if idlr_wl_nm is None:
            idlr_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/sgnl_wl_nm)
            if not silent:
                print 'Setting idler to ',idlr_wl_nm,' nm'
            RET_wl_nm = True
            new_wl_nm = idlr_wl_nm
            
        kp = self.n(pump_wl_nm)*2*np.pi/pump_wl_nm
        ks = self.n(sgnl_wl_nm)*2*np.pi/sgnl_wl_nm
        ki = self.n(idlr_wl_nm)*2*np.pi/idlr_wl_nm
        if self.length_mks is not None:
            delta_k_set_pt = delta_k_L / self.length_nm
        else:
            delta_k_set_pt = 0
        deltak = kp-ks-ki - delta_k_set_pt
        period_meter = np.pi/deltak*1.0e-9
        if not silent:
            print 'period is ',2.0*period_meter*1.0e6,' um'
        if RET_wl_nm:
            return (period_meter*2, new_wl_nm)
        else:
            return period_meter*2
