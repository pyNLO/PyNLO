# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 11:31:34 2014
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


plt.close('all')

dz = 1e-3
steps = 500
range1 = np.arange(steps)

centerwl = 835.0
fiber_length = 0.15

init = Pulse(n = 2**13)
init.gen_sech(1e4, 28.4e-3, centerwl)

fiber1 = Fiber()
fiber1.load_from_db( fiber_length, 'dudley')

evoladv  = SSFM(dz = 1e-6, local_error = 0.001, USE_SIMPLE_RAMAN = False)
yadv = np.zeros(steps)
AWadv = np.zeros((init.n, steps))
ATadv = np.copy(AWadv)

yadv, AWadv, ATadv, pulse1adv = evoladv.propagate(pulse_in = init,
                                                        fiber = fiber1,
                                                        n_steps = steps)
                                         
evolsimp = SSFM(dz = 1e-6, local_error = 0.001, USE_SIMPLE_RAMAN = True)
ysimp = np.zeros(steps)
AWsimp = np.zeros((init.n, steps))
ATsimp = np.copy(AWsimp)

ysimp, AWsimp, ATsimp, pulse1simp = evolsimp.propagate(pulse_in = init,
                                                             fiber = fiber1,
                                                             n_steps = steps)

ATmatload = np.genfromtxt('ATmat.csv',delimiter=',')
AWmatload = np.genfromtxt('AWmat.csv',delimiter=',')
Tmat  = ATmatload[0,:]
Wmat  = AWmatload[0,:]
ATmat = ATmatload[1,:]
AWmat = AWmatload[1,:]
ATmatmax = np.max(ATmat)
AWmatmax = np.max(AWmat)

ATpysimp = np.abs(ATsimp[:,-1])
AWpysimp = np.abs(AWsimp[:,-1])
ATpyadv  = np.abs(ATadv[:,-1])
AWpyadv  = np.abs(AWadv[:,-1])

ATpysimpmax = np.max(ATpysimp)
AWpysimpmax = np.max(AWpysimp)
ATpyadvmax  = np.max(ATpyadv)
AWpyadvmax  = np.max(AWpyadv)

Tpy = init.T
Wpy = init.W
wlpy = 2 * np.pi * init.c / Wpy
wlmat = 2 * np.pi * init.c / Wmat

plt.figure()
plt.subplot(211)
plt.plot(Tmat, ATmat**2 / ATmatmax**2, label = 'Dudley integrator')
plt.plot(Tpy, ATpysimp**2 / ATpysimpmax**2, 'r', 
         lw=2,alpha=0.5, label = 'Our SSFM, simple Raman')
#plt.plot(Tpy, ATpyadv**2 / ATpyadvmax**2, label = 'Our SSFM, advanced Raman')
plt.xlim(-1, 5)
plt.xlabel('Time (ps)')
plt.ylabel('Normalized Intensity')
plt.legend()

plt.subplot(212)
plt.plot(wlmat, AWmat**2 / AWmatmax**2, label = 'Dudley integrator')
plt.plot(wlpy, AWpysimp**2 / AWpysimpmax**2, 'r',
         lw=2,alpha=0.5, label = 'Our SSFM, simple Raman')
#plt.plot(wlpy, AWpyadv**2 / AWpyadvmax**2, label = 'Our SSFM, advanced Raman')
plt.xlim(400,1400)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Normalized Spectral Intensity')
plt.legend()

plt.show()