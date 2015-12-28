# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 12:42:30 2014
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
@author: dim1
"""

import numpy as np
import matplotlib.pyplot as plt
from SSFM import SSFM
from fiber import Fiber
from pulse import Pulse
#from fftw_transforms import fftcomputer as fftw
#from scipy import fftpack

plt.close('all')

steps = 100

centerwl = 850.0
gamma = 0.045
fiber_length = 0.1
P0 = 10e3
T0 = 28.4e-3

init = Pulse(n = 2**13)
init.gen_sech(P0, T0, centerwl)

fiber1 = Fiber()

fiber1.generate_fiber(fiber_length ,centerwl, [-1.276e-2, 8.119e-5, -1.321e-7,
                      3.032e-10, -4.196e-13, 2.570e-16], gamma, 0)

evol = SSFM(disable_Raman = False, disable_self_steepening = False,
            local_error = 0.1, suppress_iteration = True)

y = np.zeros(steps)
AW = np.complex64(np.zeros((init.n, steps)))
AT = np.complex64(np.copy(AW))

y, AW, AT, pulse1, = evol.propagate(pulse_in = init, fiber = fiber1, 
                                         n_steps = steps)
                                         
wl = 2 * np.pi * init.c / (init.W)

loWL = 500
hiWL = 1200
                         
iis = np.logical_and(wl>loWL,wl<hiWL)

iisT = np.logical_and(init.T>-1,init.T<5)

xW = wl[iis]
xT = init.T[iisT]
zW_in = np.transpose(AW)[:,iis]
zT_in = np.transpose(AT)[:,iisT]
zW = 10*np.log10(np.abs(zW_in)**2)
zT = 10*np.log10(np.abs(zT_in)**2)
mlIW = np.max(zW)
mlIT = np.max(zT)

D = fiber1.Beta2_to_D(init)
beta = fiber1.Beta2(init)

plt.figure()
#plt.subplot(121)
plt.pcolormesh(xW, y, zW, vmin = mlIW - 40.0, vmax = mlIW)
plt.autoscale(tight=True)
plt.xlim([loWL, hiWL])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Distance (m)')

#plt.subplot(122)
#plt.pcolormesh(xT, y, zT, vmin = mlIT - 40.0, vmax = mlIT)
#plt.autoscale(tight=True)
#plt.xlabel('Delay (ps)')
#plt.ylabel('Distance (m)')

plt.show()