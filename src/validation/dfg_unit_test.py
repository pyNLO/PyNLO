# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:54:36 2014
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
    
Based upon dfg_test.py. Verifies CW DFG by comparing numeric result against
the analytic solution (from Boyd.) Initial conditions are slightly randomized.

"""
import numpy as np
from scipy.constants import speed_of_light
from pynlo.light.DerivedPulses import CWPulse
from pynlo.light import DerivedPulses
from pynlo.media import crystals
from pynlo.interactions.ThreeWaveMixing import dfg_problem
from pynlo.util import ode_solve
from pynlo.util.ode_solve import dopr853
import unittest

class TestCWDFG(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass    
    def test_fn(self):
        npoints = 2**6
        # Physical lengths are in meters
        crystallength = 0.010
        crystal = crystals.AgGaSe2()
        crystal = crystals.AgGaSe2(length = crystallength)
        
        
        n_saves = 10
        
        # Wavelengths are in nanometers
        pump_wl_nm = 1560. + 10.0*np.random.rand()
        sgnl_wl_nm = 2200. + 10.0*np.random.rand()
        theta, idlr_wl_nm = crystal.phasematch(pump_wl_nm, sgnl_wl_nm, None, return_wavelength = True)
        crystal.set_theta(theta)
        
        
        pump_power = (1+np.random.rand()) * 5000.0
        sgnl_power = 0.10 * np.random.rand()         
        twind = 25.
        beamwaist = 100e-3
        
        pump_in = CWPulse(pump_power, pump_wl_nm, NPTS = npoints, time_window_ps = twind)
        sgnl_in = CWPulse(sgnl_power, sgnl_wl_nm, NPTS = npoints, time_window_ps = twind)
                
        integrand = dfg_problem(pump_in, sgnl_in, crystal,
                      disable_SPM = True, pump_waist = beamwaist, apply_gouy_phase = True)
        
        # Set up integrator
        rtol = 1.0e-12
        atol = 1.0e-12
        x0   = 0.0
        x1   = crystallength
        hmin = 0.0
        h1   = 0.0000001
        out  = ode_solve.Output(n_saves)
        
        # From Boyd section 2.8, also my One Note
        omega = lambda l_nm : 2.0*np.pi*speed_of_light / (l_nm*1.0e-9)
        w_1 = omega(sgnl_wl_nm)
        w_2 = omega(idlr_wl_nm)
        k_1 = np.mean(2*np.pi*crystal.n(sgnl_wl_nm, 'o') / ( sgnl_wl_nm * 1.0e-9))
        k_2 = np.mean(2*np.pi*crystal.n(idlr_wl_nm, 'o') / ( idlr_wl_nm * 1.0e-9))
        
        n_1 = crystal.n(sgnl_wl_nm, 'o')
        n_2 = crystal.n(idlr_wl_nm, 'o')
        n_3 = crystal.n(pump_wl_nm, 'mix')
        
        
        A_3 = np.sqrt(pump_power) * integrand.pump_beam.rtP_to_a(crystal.n(pump_wl_nm, 'mix'))
        A_1 = np.sqrt(sgnl_power) * integrand.sgnl_beam.rtP_to_a(crystal.n(sgnl_wl_nm, 'o'))
        
        kappa = np.mean(np.sqrt( 4 * crystal.deff**2 * w_1**2 * w_2**2 / (k_1*k_2 * speed_of_light**4)) * A_3)
        
        a = ode_solve.ODEint(integrand.ystart, x0, x1, atol, rtol, h1,hmin, out,\
                 dopr853.StepperDopr853, integrand, dtype = np.complex128)
        a.integrate()
        
        pump_out = a.out.ysave[0:a.out.count, 0         : npoints].T
        sgnl_out = a.out.ysave[0:a.out.count, npoints   :   2*npoints].T
        idlr_out = a.out.ysave[0:a.out.count, 2*npoints :   3*npoints].T
        z        = a.out.xsave[0:a.out.count]
        
        pump_power_in =    np.sum(np.abs(pump_out[:,0]), axis=0)**2
        sgnl_power_in =    np.sum(np.abs(sgnl_out[:,])**2, axis=0)
        idlr_power_in =    np.sum(np.abs(idlr_out[:,0]), axis=0)**2
        
        pump_power_out =    np.sum(np.abs(pump_out[:,-1]))**2
        sgnl_power_out =    np.sum(np.abs(sgnl_out[:,-1]))**2
        idlr_power_out =    np.sum(np.abs(idlr_out[:,-1]))**2
            
        # Compare integrated and analytic values for signal power
        numeric_sgnl    = np.sum(np.abs(sgnl_out[:, :])**2, axis=0)
        analytic_sgnl   = sgnl_power_in * np.cosh(kappa*z) **2
        self.assertLess(abs(np.sum(numeric_sgnl - analytic_sgnl) / np.sum(abs(analytic_sgnl))), 0.01 ) 
        
        # Compare integrated and analytic values for idler power
        numeric_idlr    = np.sum(np.abs(idlr_out[:, :])**2, axis=0)
        analytic_idlr   = n_1 * w_2 / (n_2*w_1) *  sgnl_power_in * np.sinh(kappa*z) **2
        self.assertLess(abs(np.sum(numeric_idlr - analytic_idlr) / np.sum(abs(analytic_idlr))), 0.02 )

class TestCW_offset_DFG(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass    
    def test_fn(self):
        npoints = 2**6
        # Physical lengths are in meters
        crystallength = 0.10
        crystal = crystals.AgGaSe2()
        crystal = crystals.AgGaSe2(length = crystallength)
        
        
        n_saves = 10
        
        # Wavelengths are in nanometers
        pump_wl_nm = 1560. + 10.0*np.random.rand()
        sgnl_wl_nm = 2200. + 10.0*np.random.rand()
        pump_freq_offset = (1-0.5*np.random.rand())
        sgnl_freq_offset = (1-0.5*np.random.rand())
        
        theta, idlr_wl_nm = crystal.phasematch(pump_wl_nm, sgnl_wl_nm, None, return_wavelength = True)
        crystal.set_theta(theta)
        
        
        pump_power = (1+np.random.rand()) * 50000.0
        sgnl_power = 0.10 * np.random.rand()         
        twind = 25.
        beamwaist = 1000e-3
        
        pump_in = CWPulse(pump_power, pump_wl_nm, NPTS = npoints, time_window_ps = twind, offset_from_center_THz = pump_freq_offset)
        sgnl_in = CWPulse(sgnl_power, sgnl_wl_nm, NPTS = npoints, time_window_ps = twind, offset_from_center_THz = sgnl_freq_offset)
                
        integrand = dfg_problem(pump_in, sgnl_in, crystal,
                      disable_SPM = True, pump_waist = beamwaist, apply_gouy_phase = False)
        
        # Set up integrator
        rtol = 1.0e-12
        atol = 1.0e-12
        x0   = 0.0
        x1   = crystallength
        hmin = 0.0
        h1   = 0.0000001
        out  = ode_solve.Output(n_saves)
        
        # From Boyd section 2.8, also my One Note
        omega = lambda l_nm : 2.0*np.pi*speed_of_light / (l_nm*1.0e-9)
        w_1 = omega(sgnl_wl_nm)
        w_2 = omega(idlr_wl_nm)
        k_1 = np.mean(2*np.pi*crystal.n(sgnl_wl_nm, 'o') / ( sgnl_wl_nm * 1.0e-9))
        k_2 = np.mean(2*np.pi*crystal.n(idlr_wl_nm, 'o') / ( idlr_wl_nm * 1.0e-9))
        
        n_1 = crystal.n(sgnl_wl_nm, 'o')
        n_2 = crystal.n(idlr_wl_nm, 'o')
        n_3 = crystal.n(pump_wl_nm, 'mix')
        
        
        A_3 = np.sqrt(pump_power) * integrand.pump_beam.rtP_to_a(crystal.n(pump_wl_nm, 'mix'))
        A_1 = np.sqrt(sgnl_power) * integrand.sgnl_beam.rtP_to_a(crystal.n(sgnl_wl_nm, 'o'))
        
        kappa = np.mean(np.sqrt( 4 * crystal.deff**2 * w_1**2 * w_2**2 / (k_1*k_2 * speed_of_light**4)) * A_3)
        
        a = ode_solve.ODEint(integrand.ystart, x0, x1, atol, rtol, h1,hmin, out,\
                 dopr853.StepperDopr853, integrand, dtype = np.complex128)
        a.integrate()
        
        pump_out = a.out.ysave[0:a.out.count, 0         : npoints].T
        sgnl_out = a.out.ysave[0:a.out.count, npoints   :   2*npoints].T
        idlr_out = a.out.ysave[0:a.out.count, 2*npoints :   3*npoints].T
        z        = a.out.xsave[0:a.out.count]
        
        pump_power_in =    np.sum(np.abs(pump_out[:,0]))**2
        sgnl_power_in =    np.sum(np.abs(sgnl_out[:,0]))**2
        idlr_power_in =    np.sum(np.abs(idlr_out[:,0]))**2
        
        pump_power_out =    np.sum(np.abs(pump_out[:,-1]))**2
        sgnl_power_out =    np.sum(np.abs(sgnl_out[:,-1]))**2
        idlr_power_out =    np.sum(np.abs(idlr_out[:,-1]))**2
            
        # Compare integrated and analytic values for signal power
        numeric_sgnl    = np.sum(np.abs(sgnl_out[:, :])**2, axis=0)
        analytic_sgnl   = sgnl_power_in * np.cosh(kappa*z) **2
        
        
        # Compare integrated and analytic values for idler power
        numeric_idlr    = np.sum(np.abs(idlr_out[:, :])**2, axis=0)
        analytic_idlr   = n_1 * w_2 / (n_2*w_1) *  sgnl_power_in * np.sinh(kappa*z) **2
        print 'idlr ',max(numeric_idlr), max(analytic_idlr)
        
        self.assertLess(abs(np.sum(numeric_sgnl - analytic_sgnl) / np.sum(abs(analytic_sgnl))), 0.01 ) 
        self.assertLess(abs(np.sum(numeric_idlr - analytic_idlr) / np.sum(abs(analytic_idlr))), 0.01 )
        
if __name__ == '__main__':
    unittest.main()