# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:23:17 2015

Sellemeier coefficients and nonlinear parameter for AsGaSe_2
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
import exceptions 

class AgGaSe2(Crystal):
    def __init__(self, theta = 0.0, **params):
        """ Load AgGaSe2 data. theta : crystal angle (radians)"""
        Crystal.__init__(self, params)
        self.mode  = 'BPM'

        self.Ao    = 6.849065
        self.Bo    = 0.417863
        self.Co    = 0.178080
        self.Do    = 1.209374
        self.Eo    = 915.345
        self.Fo    = 0.000442
        self.Go    = 0.889242
        
        self.ao    = 1.970203
        self.bo    = 0.340086
        self.co    = 1.921292
        
        self.Ae    = 6.675232
        self.Be    = 0.436579
        self.Ce    = 0.229775
        self.De    = 3.252722
        self.Ee    = 3129.32
        self.Fe    = 0.012063
        self.Ge    = 0.213957
        
        self.ae    = 1.893694
        self.be    = 4.269152
        self.ce    = 2.047204


        self.theta = theta
        self.n2    = 35e-15 / 100**2 # from Nikogosyan, originally given in cm^2 / W
        self.deff  = 28.5e-12 # from SNLO, original given in pm / V        
    def set_theta(self, angle):
        self.theta = angle
    def n(self, wl_nm, axis = "mix"):        
        """ Axis specifies crystal axis, either o, e, or mix. If mix, class
            instances value for theta sets mixing angle (0 = pure ordinary). 
            Following experimental results from Willer, Blanke, Schade
            'Difference frequency generation in AgGaSe2: sellmeier and 
            temperature-dispersion equations', use rational-exponent 
            Sellmeier from Roberts (1996) """
        wl_um = wl_nm * 1.0e-3
        no = np.sqrt(   self.Ao +
                        self.Bo/(np.power(wl_um, self.ao) - self.Co) +
                        self.Fo/(np.power(wl_um, self.bo) - self.Go) +
                        self.Do/( 1. - self.Eo / np.power(wl_um, self.co) ) )
               
        ne = np.sqrt(   self.Ae +
                        self.Be/(np.power(wl_um, self.ae) - self.Ce) +
                        self.Fe/(np.power(wl_um, self.be) - self.Ge) +
                        self.De/( 1. - self.Ee /np.power(wl_um, self.ce) ) )
               
        if axis == 'mix':
            return 1.0 / np.sqrt(np.sin(self.theta)**2/ne**2 + 
                   np.cos(self.theta)**2/no**2)
        elif axis == 'o':
            return no
        elif axis == 'e':
            return ne
        raise exceptions.ValueError("Axis was ",str(axis),"; must be 'mix', 'o', or 'e'")
    def phasematch(self, pump_wl_nm, sgnl_wl_nm, idlr_wl_nm, return_wavelength = False):
        """ Phase match mixing between pump (aligned to a mix of ne and no) and
            signal and idler (aligned to ordinary axis.)"""
        RET_WL = False
        new_wl = 0.0
        if pump_wl_nm is None:
            pump_wl_nm = 1.0/(1.0/idlr_wl_nm + 1.0/sgnl_wl_nm)
            print 'Setting pump to ',pump_wl_nm
            RET_WL = True
            new_wl = pump_wl_nm
        if sgnl_wl_nm is None:
            sgnl_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/idlr_wl_nm)
            print 'Setting signal to ',sgnl_wl_nm
            RET_WL = True
            new_wl = sgnl_wl_nm
        if idlr_wl_nm is None:
            idlr_wl_nm = 1.0/(1.0/pump_wl_nm - 1.0/sgnl_wl_nm)
            print 'Setting idler to ',idlr_wl_nm
            RET_WL = True
            new_wl = idlr_wl_nm
            
        kp_0    = 2*np.pi/pump_wl_nm
        ks      = self.n(sgnl_wl_nm, axis = 'o')*2*np.pi/sgnl_wl_nm
        ki      = self.n(idlr_wl_nm, axis = 'o')*2*np.pi/idlr_wl_nm

        n_soln  = (ks+ki) / kp_0
        n_e     = self.n(pump_wl_nm, 'e')
        n_o     = self.n(pump_wl_nm, 'o')
        print 'n_e @ pump: ',n_e, ';\t n_o @ pump: ',n_o
        a = n_e**2 - n_o**2
        b = 0.0
        c = n_o**2 - n_e**2 * n_o**2 / (n_soln**2)
        x = ( -b + np.sqrt(b**2-4*a*c) )/ (2.0  * a)
        if x < 0:
            x = ( -b - np.sqrt(b**2-4*a*c) )/ (2.0  * a)
        if np.isnan(np.arccos(x)) :
            raise exceptions.AttributeError('No phase matching condition.')
        theta = np.arccos(x)
        print 'Angle set to ',360*theta / (2.0*np.pi)
        if RET_WL and return_wavelength:
            return (theta, new_wl)
        else:
            return theta