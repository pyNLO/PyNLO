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

import numpy as np
from pynlo.util.ode_solve import dopr853
from pynlo.util import ode_solve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class integrand:
    def deriv(self, x, y, dydx):
        sigma   = 10.0
        R       = 28.0
        b       = 8.0/3.0
        dydx[0] = sigma * (y[1]- y[0])
        dydx[1] = R * y[0] - y[1] - y[0]*y[2]
        dydx[2] = -b * y[2] + y[0] * y[1]
    
ystart = np.ones((3,), dtype = np.complex128)
ystart[:] = [10.0, 1.0, 1.0]
dydxstart = np.ones((3,), dtype = np.complex128)

xstart = np.zeros((1,))

rtol = 1.0e-9
atol = 1.0e-9
x0   = 0.0
x1   = 250.0
hmin = 0.0
h1   = 0.01
out  = ode_solve.Output(5000)

a = ode_solve.ODEint(ystart, x0, x1, atol, rtol, h1,hmin, out,\
         dopr853.StepperDopr853, integrand())
a.integrate()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(a.out.ysave[:a.out.count, 0],
           a.out.ysave[:a.out.count, 1],
           a.out.ysave[:a.out.count, 2]) 
plt.show()