# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:06:10 2015
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
from pynlo.devices.grating_compressor import TreacyCompressor
import numpy as np

class TreacyTest(unittest.TestCase): 
    def setUp(self):
        pass
    def tearDown(self):
        pass    
    def test_fn(self):    
        # Check against Tom Allison's calculation
        theta = 41.5
        ruling = 1250.0
    
        gdd_check = -1.8e-24    
        tod_check = 7.793e-39
            
        TC = TreacyCompressor(ruling, theta)
        
        wl = 1060.0
        sep = 11.5e-2
        
        gdd = TC.calc_compressor_gdd(wl, sep)        
        tod = TC.calc_compressor_HOD(wl, sep, 3)
        
        self.assertLess(abs(gdd-gdd_check)/gdd_check, 0.05)
        self.assertLess(abs(tod-tod_check)/tod_check, 0.05)
        
if __name__ == '__main__':
    unittest.main()        