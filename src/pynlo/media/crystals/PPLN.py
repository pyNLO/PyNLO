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

class PPLN(Crystal):
    def __init__(self, temperature_celsius):
        self.load({'T_deg_C' : temperature_celsius})
    def load(self, params):
            """ Load AgGeSe2 data. params -- 'T' : crystal temperature (C)
            Uses parameters from Deng et al, Opt. Comm. 268, 1, 1 pp 110-114
            'Improvement to Sellmeier equation for periodically poled LiNbO3
            crystal using mid-infrared difference-frequency generation', which
            includes temperature dependence. """
    
            self.T = params["T_deg_C"]
            self.PP = 'PP'
            self.sellmeier_type ='deng'
            self.a1  = 5.39121
            self.a2  = 0.100473
            self.a3  = 0.20692
            self.a4  =  100.0
            self.a5  = 11.34927
            self.a6  =  1.544e-2
            self.b1  = 	 4.96827e-7
            self.b2  = 	 3.862e-8
            self.b3  = 	 -0.89e-8
            self.b4  = 	 2.657e-5
            self.b5  =   9.62119e-10
            self.deff=  14.9e-12 # from SNLO
            self.n2=   3e-15 / 100**2 # from Nikogosyan
            self.pp=   lambda(x): 30.49e-6 
     
    def n(self, wl):
        wl_um = wl * 1e6
        f = (self.T - 24.5)*(self.T+570.82)
        return np.sqrt(self.a1 + self.b1*f +\
                (self.a2 + self.b2*f)/(wl_um**2-(self.a3+self.b3*f)**2) +\
                (self.a4 + self.b4*f)/(wl_um**2 - self.a5**2) -\
                (self.a6 + self.b5*f) * wl_um**2)
    def calculate_poling_period(self, pump_wl, signal_wl, idler_wl):
        if pump_wl == 0:
            pump_wl = 1.0/(1.0/signal_wl + 1.0/signal_wl)
        if signal_wl == 0:
            signal_wl = 1.0/(1.0/pump_wl - 1.0/idler_wl)
        if idler_wl == 0:
            idler_wl = 1.0/(1.0/pump_wl - 1.0/signal_wl)
            
        kp = self.n(pump_wl)*2*np.pi/pump_wl
        ks = self.n(signal_wl)*2*np.pi/signal_wl
        ki = self.n(idler_wl)*2*np.pi/idler_wl
        deltak = kp-ks-ki
        period = np.pi/deltak
        print 'period is ',2.0e6*period
        return period