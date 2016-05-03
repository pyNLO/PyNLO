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
from scipy.constants import speed_of_light
from pynlo.light.DerivedPulses import CWPulse
from pynlo.media import crystals
from pynlo.interactions.ThreeWaveMixing import dfg_problem
from pynlo.util import ode_solve
from pynlo.util.ode_solve import dopr853
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.close('all')

# Test pump depetion in case where pump & signal photon counts are the same.
# Idea from Brian Washburn

npoints = 2**6
# Physical lengths are in meters
crystallength = 0.10
crystal = crystals.AgGaSe2()


n_saves = 1000

# Wavelengths are in nanometers
pump_wl_nm = 1560. 
sgnl_wl_nm = 2200. 
theta, idlr_wl_nm = crystal.phasematch(pump_wl_nm, sgnl_wl_nm, None, return_wavelength = True)
crystal.set_theta(theta)


pump_power = 2500000.0
sgnl_power = pump_power*sgnl_wl_nm / pump_wl_nm
idlr_power = 0.000

twind = 25.
beamwaist = 10e-3

pump_in = CWPulse(pump_power, pump_wl_nm, NPTS = npoints, time_window = twind, offset_from_center_THz = 0.100)
sgnl_in = CWPulse(sgnl_power, sgnl_wl_nm, NPTS = npoints, time_window = twind)

plt.plot(pump_in.wl_nm-pump_wl_nm, abs(pump_in.AW))
plt.xlim(pump_wl_nm-2,pump_wl_nm+2)
plt.show()

integrand = dfg_problem(pump_in, sgnl_in, crystal,
              disable_SPM = True, pump_waist = beamwaist)

# Set up integrator
rtol = 1.0e-8
atol = 1.0e-8
x0   = 0.0
x1   = crystallength
hmin = 0.0
h1   = 0.0000001
out  = ode_solve.Output(n_saves)

# From Boyd section 2.8, also my One Note
omega = lambda l_nm : 2.0*np.pi*speed_of_light / (l_nm*1.0e-9)
w_1 = omega(sgnl_wl_nm)
w_2 = omega(idlr_wl_nm)
k_1 = 2*np.pi*crystal.n(sgnl_wl_nm, 'o') / ( sgnl_wl_nm * 1.0e-9)
k_2 = 2*np.pi*crystal.n(idlr_wl_nm, 'o') / ( idlr_wl_nm * 1.0e-9)

n_1 = crystal.n(sgnl_wl_nm, 'o')
n_2 = crystal.n(idlr_wl_nm, 'o')
n_3 = crystal.n(pump_wl_nm, 'mix')


A_3 = np.sqrt(pump_power) * integrand.pump_beam.rtP_to_a(crystal.n(pump_wl_nm, 'mix'))
A_1 = np.sqrt(sgnl_power) * integrand.sgnl_beam.rtP_to_a(crystal.n(sgnl_wl_nm, 'o'))

kappa = np.mean(np.sqrt( 4 * np.mean(crystal.deff)**2 * w_1**2 * w_2**2 / (k_1*k_2 * speed_of_light**4)) * A_3)

a = ode_solve.ODEint(integrand.ystart, x0, x1, atol, rtol, h1,hmin, out,\
         dopr853.StepperDopr853, integrand, dtype = np.complex128)
print 'Running'
a.integrate()
print 'integrated!'
print 'coupling length is ',1.0/kappa,'m'
print 'calculated coupling coeff. is ',kappa

print 'theoretical d idler/dz is ',A_1*A_3*(2*w_2**2*crystal.deff)/(k_2*speed_of_light**2)

pump_out = a.out.ysave[0:a.out.count, 0         : npoints].T
sgnl_out = a.out.ysave[0:a.out.count, npoints   :   2*npoints].T
idlr_out = a.out.ysave[0:a.out.count, 2*npoints :   3*npoints].T
z        = a.out.xsave[0:a.out.count]

pump_power_in =    np.sum(np.abs(pump_out[:,0]))**2
sgnl_power_in =    np.sum(np.abs(sgnl_out[:,0]))**2
idlr_power_in =    np.sum(np.abs(idlr_out[:,0]))**2

pump_power_out =    np.sum(np.abs(pump_out[:,-1]))**2
sgnl_power_out =    np.sum(np.abs(sgnl_out[:,-1]))**2
idlr_power_out =    np.sum(np.abs(idlr_out[:,-1]))**2
    

                        
print "pump power in: ",    pump_power_in*1000, "mW"                           
print "signal power in: ",  sgnl_power_in*1000, "mW"                           
print "idler power in: ",   idlr_power_in*1000, "mW"                           
print "pump power out: ",   pump_power_out*1000, "mW"                           
print "signal power out: ", sgnl_power_out*1000, "mW"                           
print "idler power out: ",  idlr_power_out*1000, "mW"    

plt.figure()
plt.plot(z, np.sum(np.abs(pump_out[:, :])**2, axis=0))
plt.plot(z, np.sum(np.abs(sgnl_out[:, :])**2, axis=0), label = 'Model')
plt.plot(z, np.sum(np.abs(idlr_out[:, :])**2, axis=0))
plt.xlabel('Distance (meters)')
plt.ylabel('Power (W)')
plt.show()

#
#pump_intensity = np.sum(np.abs(pump_out)**2, axis=0)
#sgnl_intensity = np.sum(np.abs(sgnl_out)**2, axis=0)
#idlr_intensity = np.sum(np.abs(idlr_out)**2, axis=0)
#
#plt.plot(z*1000, pump_intensity , color = 'g', linewidth = 1)
#plt.plot(z*1000, (sgnl_intensity), color = 'r', linewidth = 1)
#plt.plot(z*1000, (idlr_intensity), color = 'k', linewidth = 1)
#plt.xlabel('Propagation length (mm)')
#plt.ylabel('Power (mW)')
#plt.show()