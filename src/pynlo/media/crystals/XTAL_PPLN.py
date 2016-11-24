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
from scipy import interpolate
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
    # Data for LiNb absorption. From Leidinger (Optics Express, 2015)
    # Units here are per centimeter (converted below to per meter)
    alpha_data = np.array([[  1.47041521e+02,   3.68364978e+00],
       [  1.57140656e+02,   8.68029576e-01],
       [  1.61629161e+02,   1.85667130e+00],
       [  1.66117666e+02,   4.00766264e-01],
       [  1.80705306e+02,   1.71628970e-01],
       [  1.95292946e+02,   8.50726840e-02],
       [  2.09880586e+02,   4.86043881e-02],
       [  2.19979721e+02,   2.67443612e-02],
       [  2.34567362e+02,   1.76855134e-02],
       [  2.87307291e+02,   1.32565882e-02],
       [  3.45657852e+02,   8.58510805e-03],
       [  4.08496917e+02,   5.09283618e-03],
       [  4.71335982e+02,   3.61566390e-03],
       [  5.00511262e+02,   2.90967882e-03],
       [  5.15098903e+02,   2.35134492e-03],
       [  5.34175047e+02,   1.36602560e-03],
       [  5.63350328e+02,   9.69811177e-04],
       [  6.07113248e+02,   5.24791485e-04],
       [  6.17212384e+02,   6.77108673e-04],
       [  6.31800024e+02,   4.01672702e-04],
       [  6.89028458e+02,   3.93369674e-04],
       [  7.51867523e+02,   3.47034156e-04],
       [  8.05729579e+02,   3.34228612e-04],
       [  8.49492499e+02,   3.66402115e-04],
       [  9.16820069e+02,   3.23243141e-04],
       [  1.04810883e+03,   2.74645234e-04],
       [  1.14910019e+03,   3.23243141e-04],
       [  1.28038895e+03,   2.46377230e-04],
       [  1.32864037e+03,   2.02455234e-04],
       [  1.35781565e+03,   1.65669729e-04],
       [  1.45992913e+03,   1.05511551e-04],
       [  1.52276820e+03,   9.15406939e-05],
       [  1.60468341e+03,   9.30832106e-05],
       [  1.65854546e+03,   9.46517195e-05],
       [  1.77412446e+03,   9.30832106e-05],
       [  1.92000086e+03,   1.22123742e-04],
       [  2.01650371e+03,   2.25683864e-04],
       [  2.10402955e+03,   3.60330329e-04],
       [  2.12759420e+03,   7.00120323e-04],
       [  2.13208270e+03,   1.58109853e-03],
       [  2.13208270e+03,   2.43125576e-03],
       [  2.17584563e+03,   3.61566390e-03],
       [  2.22409705e+03,   2.14487503e-03],
       [  2.24878383e+03,   9.03325523e-04],
       [  2.33069904e+03,   2.26458040e-03],
       [  2.33069904e+03,   4.92544415e-03],
       [  2.36997345e+03,   1.39964377e-02],
       [  2.38007259e+03,   3.70464612e-02],
       [  2.40363724e+03,   1.93879556e-02],
       [  2.43730102e+03,   7.57385718e-03],
       [  2.44291165e+03,   2.71020504e-03],
       [  2.51023922e+03,   5.67716095e-03],
       [  2.57307829e+03,   8.44284102e-03],
       [  2.62694034e+03,   5.77282458e-03],
       [  2.65050499e+03,   8.58510805e-03],
       [  2.67968027e+03,   5.37706711e-03],
       [  2.73354233e+03,   2.75587365e-03],
       [  2.74812997e+03,   8.91403555e-03],
       [  2.77169462e+03,   3.84658493e-02],
       [  2.78628226e+03,   1.16950777e-02],
       [  2.80086990e+03,   6.22365562e-03],
       [  2.83004518e+03,   3.49682376e-03],
       [  2.86931959e+03,   5.28796173e-03],
       [  2.93664716e+03,   8.58510805e-03],
       [  2.96133394e+03,   1.03174995e-02],
       [  2.99948623e+03,   9.77211812e-03],
       [  3.04773765e+03,   7.70148130e-03],
       [  3.10159971e+03,   6.43516758e-03],
       [  3.15433964e+03,   5.28796173e-03],
       [  3.21269020e+03,   5.28796173e-03],
       [  3.25645312e+03,   5.00844090e-03],
       [  3.29460541e+03,   6.79431396e-03],
       [  3.33387983e+03,   8.44284102e-03],
       [  3.38213125e+03,   6.09500563e-03],
       [  3.45506945e+03,   8.13130058e-03],
       [  3.51790852e+03,   1.12635299e-02],
       [  3.57625908e+03,   1.45326934e-02],
       [  3.65368578e+03,   1.67506599e-02],
       [  3.68734957e+03,   2.16124259e-02],
       [  3.74008950e+03,   3.20071226e-02],
       [  3.83210384e+03,   5.21817193e-02],
       [  3.91962968e+03,   8.19335059e-02],
       [  4.00715552e+03,   1.43408929e-01],
       [  4.08907073e+03,   2.16865642e-01],
       [  4.15752043e+03,   2.69484476e-01],
       [  4.18557358e+03,   2.94193999e-01],
       [  4.23943564e+03,   2.94193999e-01],
       [  4.27309943e+03,   2.55239569e-01],
       [  4.32135085e+03,   2.32827168e-01],
       [  4.39428905e+03,   2.74025454e-01],
       [  4.51099017e+03,   4.54276018e-01],
       [  4.64564531e+03,   6.64385371e-01],
       [  4.79601022e+03,   7.13284793e-01],
       [  4.98564954e+03,   9.01287026e-01],
       [  5.06307625e+03,   1.09681827e+00],
       [  5.17865524e+03,   1.51931973e+00],
       [  5.23251730e+03,   2.06969583e+00],
       [  5.28525723e+03,   2.61521067e+00],
       [  5.34360779e+03,   4.10629204e+00],
       [  5.36717244e+03,   5.18859757e+00],
       [  5.42103449e+03,   4.48280545e+00],
       [  5.46479741e+03,   4.10629204e+00],
       [  5.49397269e+03,   4.55834346e+00],
       [  5.53773561e+03,   4.10629204e+00],
       [  5.57139940e+03,   3.88923409e+00],
       [  5.59496405e+03,   4.81274438e+00],
       [  5.62974996e+03,   5.08134339e+00],
       [  5.66790225e+03,   4.40851921e+00],
       [  5.70717666e+03,   4.24584505e+00],
       [  5.73074131e+03,   4.55834346e+00],
       [  5.79919101e+03,   4.10629204e+00],
       [  5.87100708e+03,   4.31739014e+00],
       [  5.94843379e+03,   4.65455844e+00]])
    alpha_interp = None
    def __init__(self, T, **params):
        Crystal.__init__(self, params)
        self.load(T)
        self.alpha_interp = interpolate.interp1d(self.alpha_data[:,0],
                                                 self.alpha_data[:,1] * 1e2, 
                                                 fill_value=10, 
                                                 bounds_error=False)
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
    def alpha(self, wavelength_nm):
        """
        Return interpolated value of linear absorption coefficient for LiNb at
        wavelengths specified.
        
        Parameters
        ----------
        wavelength_nm : float
        
        Returns
        -------
        alpha : float or array
            Absorption coefficient (m^-1)
        """
        return self.alpha_interp(wavelength_nm)