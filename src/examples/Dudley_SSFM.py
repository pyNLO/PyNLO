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

# This script simulates supercontinuum generation in a silica fiber.
# It basically reproduces Fig. 3 from 
# Dudley, Gentry, and Cohen: Supercontinuum generation in photonic crystal fiber,
# Rev. Mod. Phys., Vol. 78, No. 4, October-December 2006
import numpy as np
import matplotlib.pyplot as plt
from pynlo.interactions.FourWaveMixing import SSFM
from pynlo.media.fibers import fiber
from pynlo.light.DerivedPulses import SechPulse

dz = 1e-3
steps = 100
range1 = np.arange(steps)

centerwl = 835.0
fiber_length = 0.15

pump_power = 1.0e4 # Peak power
pump_pulse_length = 28.4e-3

npoints = 2**13

init = SechPulse(power                  =   pump_power, 
                 T0_ps                  =   pump_pulse_length, 
                 center_wavelength_nm   =   centerwl, 
                 time_window_ps         = 10.0,
                 GDD = 0, TOD = 0.0, 
                 NPTS = npoints, 
                 frep_MHz               = 100.0, 
                 power_is_avg           = False)

fiber1 = fiber.FiberInstance() 
fiber1.load_from_db( fiber_length, 'dudley')

evol = SSFM.SSFM(dz = dz, local_error = 0.001, USE_SIMPLE_RAMAN = True)
y = np.zeros(steps)
AW = np.zeros((init.NPTS, steps))
AT = np.copy(AW)

y, AW, AT, pulse1 = evol.propagate(pulse_in = init, fiber = fiber1, 
                                   n_steps = steps)
                           
wl = init.wl_nm

loWL = 400
hiWL = 1400
                         
iis = np.logical_and(wl>loWL,wl<hiWL)

iisT = np.logical_and(init.T_ps>-1,init.T_ps<5)

xW = wl[iis]
xT = init.T_ps[iisT]
zW_in = np.transpose(AW)[:,iis]
zT_in = np.transpose(AT)[:,iisT]
zW = 10*np.log10(np.abs(zW_in)**2)
zT = 10*np.log10(np.abs(zT_in)**2)
mlIW = np.max(zW)
mlIT = np.max(zT)

D = fiber1.Beta2_to_D(init)
beta = fiber1.Beta2(init)

plt.figure()
plt.subplot(121)
plt.plot(wl,D,'x')
plt.xlim(400,1600)
plt.ylim(-400,300)
plt.xlabel('Wavelength (nm)')
plt.ylabel('D (ps/nm/km)')
plt.subplot(122)
plt.plot(wl,beta*1000,'x')
plt.xlim(400,1600)
plt.ylim(-350,200)
plt.xlabel('Wavelength (nm)')
plt.ylabel(r'$\beta_2$ (ps$^2$/km)')

plt.figure()
plt.subplot(121)
plt.pcolormesh(xW, y, zW, vmin = mlIW - 40.0, vmax = mlIW)
plt.autoscale(tight=True)
plt.xlim([loWL, hiWL])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Distance (m)')

plt.subplot(122)
plt.pcolormesh(xT, y, zT, vmin = mlIT - 40.0, vmax = mlIT)
plt.autoscale(tight=True)
plt.xlabel('Delay (ps)')
plt.ylabel('Distance (m)')

plt.show()