# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 08:50:30 2015
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

import math
from scipy import optimize

from pynlo.media.fibers import fiber
from pynlo.media.fibers.calculators import DTabulationToBetas
###############################################################################
""" 
Solve the dispersive wave resonance equation, (2) from "Dispersive wave 
blue-shift in supercontinuum generation", Dane R. Austin, C. Martijn de Sterke,
Benjamin J. Eggleton, Thomas G. Brown
"""
###############################################################################
## Fiber parameters
fiber1 = fiber.FiberInstance()
fiber1.fiberloader.print_fiber_list()
fibername = 'PMHNLF_2_2_FASTAXIS_LOWER_D'
fiber1.load_from_db( 1, fibername)

center_wavelength_nm = 1560.0
poly_order = 2

betas, omegaAxis, data, fit = DTabulationToBetas(center_wavelength_nm,
                           np.transpose(np.vstack((fiber1.x,fiber1.y))),
                            poly_order,
                            DDataIsFile = False,
                            return_diagnostics = True)
plt.figure(figsize = (12, 6))
plt.title(fibername)
plt.subplot(121)                            
plt.plot(omegaAxis / (2.0*np.pi), data* 1.0e6, label = 'OFS Data' )
plt.plot(omegaAxis / (2.0*np.pi), fit* 1.0e6, label = 'Fit' )
plt.ylabel('GVD (fs^2 / m)')
plt.xlabel('Frequency from 1560 nm (THz)')
plt.legend(loc=2)
plt.subplot(122)  
plt.plot(omegaAxis / (2.0*np.pi), (data - fit)* 1.0e6)
plt.ylabel('Fit Residuals (fs^2 / m)')
plt.xlabel('Frequency from 1560 nm (THz)')

###############################################################################
## Pulse parameters
# Calculate P0 from frep, Pavg, and pulse length

Pavg = 200.0e-3
fr   = 160.0e6
t0   = 100e-15

EPP  = Pavg / fr

P0 = 0.94 * EPP / t0 # Gaussian pulse

###############################################################################
## Solve
def fn(x):
    eqn = 0
    for n in xrange(len(betas)):
       print betas[n] * np.power(x, n+2) / math.factorial(n+2)
       eqn += betas[n] * np.power(x, n+2) / math.factorial(n+2)
    eqn -= fiber1.gamma * P0 / 2.0
    return abs(eqn)

result = optimize.minimize(fn, -betas[0]/betas[1])