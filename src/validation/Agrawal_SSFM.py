# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 15:39:12 2014
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
from pynlo.interactions import FourWaveMixing
from pynlo.media.fibers import fiber
from pynlo.light.DerivedPulses import SechPulse

plt.close('all')

steps = 100

centerwl = 1550.0
gamma = 2e-3

fiber_length = 2500.0
P0 = 10.0
T0 = 0.1
nPoints = 2**13

pulse = SechPulse(P0, T0, centerwl, NPTS = nPoints)

fiber1 = fiber.FiberInstance() 
fiber1.generate_fiber(fiber_length,centerwl, [0,0], gamma, 0)

evol = FourWaveMixing.SSFM.SSFM(disable_Raman = True, disable_self_steepening = True,
            local_error = 0.1, suppress_iteration = True)


y = np.zeros(steps)
AW = np.complex64(np.zeros((pulse.NPTS, steps)))
AT = np.complex64(np.copy(AW))


y, AW, AT, pulse1, = evol.propagate(pulse_in = pulse, fiber = fiber1, 
                                         n_steps = steps)

m = 1
T = pulse.T_ps
Leff = fiber_length
LNL = 1/(gamma * P0)
dw_T = (2*m/(T0 * 1e-12)) * (Leff/LNL) * (T/T0)**(2*m-1) * np.exp(-(T/T0)**(2*m))                             
                             
wl = 1e9 * 2 * np.pi * 3e8 / (pulse.W_THz * 1e12)

loWL = 1200
hiWL = 2000

print wl
                         
iis = np.logical_and(wl>loWL,wl<hiWL)
iisT = np.logical_and(pulse.T_ps>-1,pulse.T_ps<5)

xW = wl[iis]
xT = pulse.T_ps[iisT]
zW_in = np.transpose(AW)[:,iis]
zT_in = np.transpose(AT)[:,iisT]
zW = 10*np.log10(np.abs(zW_in)**2)
zT = 10*np.log10(np.abs(zT_in)**2)
mlIW = np.max(zW)
mlIT = np.max(zT)

plt.figure()
plt.plot(T/T0,dw_T)
plt.xlim(-2,2)
x = (pulse.W_THz - pulse.center_frequency_THz) / (2* np.pi) * T0

plt.figure()
plt.pcolormesh(x, y[0:-1] / LNL, 10 * np.log10(np.abs(np.transpose(AW))**2), vmin = mlIW - 40.0, vmax = mlIW,
               cmap = plt.cm.gray)
plt.xlim(-8, 8)
plt.xlabel(r'($\nu - \nu_0) \times T_0$')
plt.ylabel(r'Distance ($z/L_{NL})$')


plt.show()