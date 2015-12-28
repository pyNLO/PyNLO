# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:54:36 2014
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
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pynlo.light.pulseclass import Pulse
from pynlo.media.crystals.PPLN import PPLN
from pynlo.interactions.ThreeWaveMixing import dfg_problem
from pynlo.util import ode_solve
from pynlo.util.ode_solve import dopr853

from gnlse_ffts import IFFT_t
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.close('all')

npoints = 2**6
crystallength = 0.46*1e-3
crystal = PPLN(45, length = crystallength)

n_saves = 500

pump_wl = 1064.
pump_power = 100.0

twind = 25.
beamwaist = 10e-6

crystal.set_pp(crystal.calculate_poling_period(pump_wl*1e-9*0.5, pump_wl*1e-9,  0))

pump_in = Pulse(n = npoints, units = 'mks')
pump_in.gen_CW(0.0, pump_wl/2.0, time_window = twind)

sgnl_in = Pulse(n = npoints, units = 'mks')
idlr_in = Pulse(n = npoints, units = 'mks')

sgnl_in.gen_CW(pump_power/2.0, pump_wl , time_window = twind)
idlr_in.gen_CW(pump_power/2.0, pump_wl , time_window = twind)

integrand = dfg_problem(pump_in, sgnl_in, idlr_in, crystal,
              disable_SPM = True, waist = beamwaist)

# Set up integrator
rtol = 1.0e-6
atol = 1.0e-6
x0   = 0.0
x1   = crystallength
hmin = 0.0
h1   = 0.00001
out  = ode_solve.Output(n_saves)

a = ode_solve.ODEint(integrand.ystart, x0, x1, atol, rtol, h1,hmin, out,\
         dopr853.StepperDopr853, integrand)
a.integrate()

print 'integrated!'

pump_out = a.out.ysave[0:a.out.count, 0         : npoints].T
sgnl_out = a.out.ysave[0:a.out.count, npoints   :   2*npoints].T
idlr_out = a.out.ysave[0:a.out.count, 2*npoints :   3*npoints].T
z        = a.out.xsave[0:a.out.count]

pump_power_in =    np.round(1e3 * np.trapz(abs(IFFT_t(pump_out[:,0]))**2,
                            pump_in.T) * pump_in.frep, decimals = 4)
signal_power_in =  np.round(1e3 * np.trapz(abs(IFFT_t(sgnl_out[:,0]))**2,
                            sgnl_in.T) * sgnl_in.frep, decimals = 4)
idler_power_in =   np.round(1e3 * np.trapz(abs(IFFT_t(idlr_out[:,0]))**2,
                            idlr_in.T) * idlr_in.frep, decimals = 4)
pump_power_out =   np.round(1e3 * np.trapz(abs(IFFT_t(pump_out[:,-1]))**2,
                            pump_in.T) * sgnl_in.frep, decimals = 4)
signal_power_out = np.round(1e3 * np.trapz(abs(IFFT_t(sgnl_out[:,-1]))**2,
                            sgnl_in.T) * sgnl_in.frep, decimals = 4)
idler_power_out =  np.round(1e3 * np.trapz(abs(IFFT_t(idlr_out[:,-1]))**2,
                            idlr_in.T) * sgnl_in.frep, decimals = 4)
                            
print "pump power in: ",    pump_power_in, "mW"                           
print "signal power in: ",  signal_power_in, "mW"                           
print "idler power in: ",   idler_power_in, "mW"                           
print "pump power out: ",   pump_power_out, "mW"                           
print "signal power out: ", signal_power_out, "mW"                           
print "idler power out: ",  idler_power_out, "mW"    

plt.figure()
for x in xrange(len(pump_out[:,0])):
    plt.plot(np.abs(sgnl_out[:, x]))
plt.show()

pump_intensity = np.sum(np.abs(pump_out)**2, axis=0)
sgnl_intensity = np.sum(np.abs(sgnl_out)**2, axis=0)
idlr_intensity = np.sum(np.abs(idlr_out)**2, axis=0)

plt.plot(z*1000, pump_intensity , color = 'g', linewidth = 1)
plt.plot(z*1000, (sgnl_intensity), color = 'r', linewidth = 1)
plt.plot(z*1000, (idlr_intensity), color = 'k', linewidth = 1)
plt.xlabel('Propagation length (mm)')
plt.ylabel('Power (mW)')
plt.show()