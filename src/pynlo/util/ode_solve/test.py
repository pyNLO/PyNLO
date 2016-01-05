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
from pynlo.util.ode_solve import dopr853
from pynlo.util import ode_solve

class integrand:
    def deriv(self, x, y, dydx) :
        # A Bernoulli equation, see Handbook of Mathematical Formulas and Integrals
        # pp 349
        dydx[0] = (-6.0 * y[0] + 3 * x*np.power(y[0], 4./3.)) / x

class IntegratorTest(unittest.TestCase):
    
    def test(self):
        # With initial condition y(1) = 2, solution is
        def exact(x):
            return np.power(x+x**2*(0.5*2**(2./3.)-1),-3)
        
        ystart = np.ones((1,))
        ystart[:] = [2.0]        
        
        rtol = 1.0e-13
        atol = 1.0e-13
        x0   = 1.0
        x1   = 4.0
        hmin = 0.0
        h1   = 0.01
        out  = ode_solve.Output(1)
        
        a = ode_solve.ODEint(ystart, x0, x1, atol, rtol, h1,hmin, out,\
                 dopr853.StepperDopr853, integrand() )
        a.integrate()
        y_calc = a.out.ysave[a.out.count-1]
        y_exct = exact(a.out.xsave[a.out.count-1])
        self.assertAlmostEqual(y_calc, y_exct, delta = 2*atol*y_exct)
if __name__ == '__main__':
    unittest.main()