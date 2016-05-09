# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 13:29:08 2014
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

fiber1 = Fiber()

steps = 250

centerwl = 1550.0
gamma = 2e-3
fiber_length = 100.0
T0 = 50e-3
D = 4 # ps / km / nm
beta2 = -2 * np.pi * fiber1.c / centerwl**2 * D
beta3 = 0.1
betas = [beta2, beta3]
print betas
P0 = abs(betas[0] * 1e-3) / gamma / T0**2
init = Pulse(n = 2**14)
init.gen_sech(P0, T0, centerwl, time_window=25)
fiber1.generate_fiber(fiber_length, centerwl, betas, gamma, 0, "ps^n/km")

evol = SSFM(disable_Raman = False, disable_self_steepening = False,
            local_error = 0.001, suppress_iteration = True, USE_SIMPLE_RAMAN = True)

y = np.zeros(steps)
AW = np.complex64(np.zeros((init.n, steps)))
AT = np.complex64(np.copy(AW))
wl = 2 * np.pi * init.c / (init.W)

shift = np.zeros((len(y), 3))
fwhm = np.copy(shift)

chirps = np.array([0.0])*-1
j = 0

for chirp in chirps:
    
    init.gen_sech(P0, T0, centerwl, chirp2 = chirp, time_window=25)
    
    y, AW, AT, pulse1 = evol.propagate(pulse_in = init, fiber = fiber1, 
                                             n_steps = steps)    

    for each in range(len(y) - 1):
        peak = np.argmax(abs(AW[:,each]))
        shift[each, j] = init.c/centerwl - init.W[peak] / (2 * np.pi)
        i = abs(AT[:,each])**2 > np.max(abs(AT[:,each]))**2/2
        fwhm[each, j] = init.dT * sum(i) * 1e3 / 1.76
    
    j += 1

plt.figure()
plt.subplot(121)
for plots in range(len(chirps)):
    plt.plot(y[:-1], shift[:,plots])
plt.autoscale(tight=True)
plt.xlabel("Distance (m)")
plt.ylabel("RIFS (THz)")

plt.subplot(122)
for plots in range(len(chirps)):
    plt.plot(y[:-1], fwhm[:,plots])
plt.xlabel("Distance (m)")
plt.ylabel("pulse width (fs)")

loWL = 1500
hiWL = 1600
                         
iis = np.logical_and(wl>loWL,wl<hiWL)

iisT = np.logical_and(init.T>-5,init.T<5)

xW = wl[iis]
xT = init.T[iisT]
zW_in = np.transpose(AW)[:,iis]
zT_in = np.transpose(AT)[:,iisT]
if False:
    zW = 10*np.log10(np.abs(zW_in)**2)
    zT = 10*np.log10(np.abs(zT_in)**2)
else:
    zW = np.abs(zW_in)**2
    zT = np.abs(zT_in)**2

mlIW = np.max(zW)
mlIT = np.max(zT)

#D = fiber1.Beta2_to_D(init)
#beta = fiber1.Beta2(init)
#
#x = (init.W - init.w0) / (2* np.pi) * T0
#b2 = beta[0] # in ps^2 / m
#LD = T0**2 / abs(b2)
#ynew = y / LD

#plt.figure()
#plt.pcolormesh(x, ynew, 10*np.log10(np.abs(np.transpose(AW))**2),
#               vmin = mlIW - 20.0, vmax = mlIW, cmap =  plt.cm.gray)
#plt.autoscale(tight=True)
#plt.xlim([-4, 4])
#plt.xlabel('(v - v0) T0')
#plt.ylabel('z/LD')

plt.figure()
plt.subplot(221)
plt.pcolormesh(xW, y, zW)#, vmin = mlIW - 40.0, vmax = mlIW)
plt.autoscale(tight=True)
plt.xlim([loWL, hiWL])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Distance (m)')

plt.subplot(222)
plt.pcolormesh(xT, y, zT)#, vmin = mlIT - 40.0, vmax = mlIT)
plt.autoscale(tight=True)
plt.xlabel('Delay (ps)')
plt.ylabel('Distance (m)')


plt.subplot(223)
plt.plot(init.T, abs(AT[:,steps-1])**2)
plt.plot(init.T, abs(AT[:,0])**2)
#plt.xlim(-3,3)

plt.subplot(224)
plt.plot(init.wl, abs(AW[:,steps-1])**2)
plt.plot(init.wl, abs(AW[:,0])**2)
plt.xlim(1500,1630)
plt.show()