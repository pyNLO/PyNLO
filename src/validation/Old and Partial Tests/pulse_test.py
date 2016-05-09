# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:19:43 2015
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
import unittest
import numpy as np
from pynlo.light.DerivedPulses import SechPulse, CWPulse


class SechPowerTest(unittest.TestCase):    
    TEST_PTS = 2**8
    power = 3.2
    fr = 2720.
    EPP = power / (fr*1e6)
    def setUp(self):        
        self.sech = SechPulse(power                 =   self.power,
                              T0_ps                 = 0.0100, 
                              center_wavelength_nm  =   1064, 
                              NPTS                  = self.TEST_PTS,
                              time_window_ps        = 12.345, 
                              frep_MHz              = self.fr, 
                              power_is_avg          = True)
    def tearDown(self):
        pass
    def test_wavelength_meters(self):   
        self.assertAlmostEqual(self.sech.calc_epp(), self.EPP)
class SechTest(unittest.TestCase):    
    TEST_PTS = 2**8
    def setUp(self):
        self.sech = SechPulse(power                 = 2.727, 
                              T0_ps                 = 0.0100, 
                              center_wavelength_nm  = 1064, 
                              NPTS                  = self.TEST_PTS, 
                              time_window_ps        = 12.345)
    def tearDown(self):
        pass
    def test_wavelength_meters(self):   
        self.assertAlmostEqual(self.sech.wl_mks[int(self.sech.NPTS/2)], 1064.*1e-9)
    def test_wavelength_nm(self): 
        self.assertAlmostEqual(self.sech.wl_nm[int(self.sech.NPTS/2)], 1064.)
    def test_frequency_Hz(self):        
        self.assertAlmostEqual(self.sech.W_mks[int(self.sech.NPTS/2)] /\
            ( 2*np.pi*299792458/(1064.*1e-9)), 1.0)
    def test_frequency_THz(self):  
        self.assertAlmostEqual(self.sech.W_THz[int(self.sech.NPTS/2)] /\
            ( 1e-12*2*np.pi*299792458/(1064.*1e-9)), 1.0)
    def test_npts(self):
        self.assertEqual(self.sech.NPTS, self.TEST_PTS)
        self.assertEqual(self.sech._n,   self.TEST_PTS)
    def test_timewindow(self):
        self.assertAlmostEqual(self.sech.time_window_ps, 12.345)
        self.assertAlmostEqual(self.sech.time_window_mks, 12.345e-12)
    def test_timeaxis(self):
        self.assertAlmostEqual(self.sech.T_ps[-1] - self.sech.T_ps[0], 12.345,1)
        self.assertAlmostEqual(self.sech.T_mks[-1] - self.sech.T_mks[0], 12.345e-12,1) 
        
    def test_temporal_peak(self):        
        self.assertAlmostEqual(np.max(np.abs(self.sech.AT))**2.0/ 2.727, 1, 2)
        
    def test_temporal_width(self):
        Tfwhm = 2.0*np.arccosh( np.sqrt(2.0)) * 0.0100
        half_max = 0.5*np.max(np.abs(self.sech.AT)**2)
        T1 = max(self.sech.T_ps[np.abs(self.sech.AT)**2 >= half_max])
        T2 = min(self.sech.T_ps[np.abs(self.sech.AT)**2 >= half_max])
        self.assertTrue( abs( (T1-T2) - Tfwhm) < 2*self.sech.dT_ps)
        
class CWTest(unittest.TestCase):    
    TEST_PTS = 2**8
    def setUp(self):
        self.cw = CWPulse(1, 1550, NPTS = self.TEST_PTS)
    def tearDown(self):
        pass
    def test_wavelength_meters(self):
        center_wl = self.cw.wl_nm[np.argmax(abs(self.cw.AW))]
        self.assertAlmostEqual(center_wl, 1550)
if __name__ == '__main__':
    unittest.main()