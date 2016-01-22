# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:23:17 2015

Sellemeier coefficients and nonlinear parameter for AsGaS
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

        self.Ao    = 5.814100
        self.Bo    = 0.0867547
        self.Co    = 0.0356502
        self.Do    = 176380.0
        self.Eo    = 112586195.0
        self.Fo    = 0.0821721
        self.Go    = -0.315646
        self.Jo    = 0.506566
        self.Ko    = -6.582197
        
        self.ao    = 3.156983
        self.bo    = 4.430430
        self.co    = 6.604280
        self.do    = 2.225043
        
        self.Ae    = 5.530050
        self.Be    = 0.0510941
        self.Ce    = 0.141109
        self.De    = 4253.78
        self.Ee    = 4304924.0
        self.Fe    = 0.195314
        self.Ge    = -0.0910735
        self.Je    = 0.
        self.Ke    = 0.
        
        self.ae    = 2.359877
        self.be    = 2.566664
        self.ce    = 0.
        self.de    = 2.383834


        self.theta = theta
        self.n2    = 0.0      # no data easily found...
        self.deff  = 8.69e-12 # from SNLO, original given in pm / V        
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
                        self.Jo/(np.power(wl_um, self.co) - self.Ko) +
                        self.Do/( 1. - self.Eo / np.power(wl_um, self.do) ) )
               
        ne = np.sqrt(   self.Ae +
                        self.Be/(np.power(wl_um, self.ae) - self.Ce) +
                        self.Fe/(np.power(wl_um, self.be) - self.Ge) +
                        self.Je/(np.power(wl_um, self.ce) - self.Ke) +
                        self.De/( 1. - self.Ee /np.power(wl_um, self.de) ) )
               
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