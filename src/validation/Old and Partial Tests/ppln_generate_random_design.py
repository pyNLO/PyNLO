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

from pynlo.media.crystals.XTAL_PPLN import PPLN
from scipy import integrate

plt.close('all')

npoints = 2**6
crystallength = 40*1e-3
crystal = PPLN(45, length = crystallength)


pump_wl = 1064.
crystal.set_pp(crystal.calculate_poling_period(pump_wl, 1540, None))
sgnl_stop_wl = 1700
NPTS = 1000
mix_bw =  crystal.calculate_mix_phasematching_bw(1064, np.linspace(1300, sgnl_stop_wl,NPTS))
idler =   1.0/(1.0/1064 - 1.0/np.linspace(1300, sgnl_stop_wl,NPTS))

print crystal.invert_dfg_qpm_to_signal_wl(1064, 24e-6)

# ODE for finding 'ideal' QPM structure
# dLambda/dz = 1/phasematching BW
# scale = 4.65e-9 # for propto BW
#scale = 1.3e5 # for propto 1/BW
scale = 7e-6 / (1e3*crystallength) # for linear chirp 10 um / crystal length
def dLdz(L, z):
    signal = crystal.invert_dfg_qpm_to_signal_wl(pump_wl, L)
    bw = crystal.calculate_mix_phasematching_bw(pump_wl, signal)
    #return 1.0/(scale*bw)
    #return (scale*bw)
    return scale

z = 0
L = 32e-6 # perid to start at
period_len = L
print("Begin APPLN design")
design = [ [z, L] ]
z = period_len/2.0
while z < 5e-3:
    signal = ( np.random.rand() * (5200-3000) + 3000 )
    bw_invm_m = crystal.calculate_mix_phasematching_bw(pump_wl, signal)
    print ("Signal: %f nm"%(signal))        
    L = crystal.calculate_poling_period(pump_wl, signal, None)[0]
    z += L
    design.append([z+L,L])    
    
design = np.array(design)
print design

grating_zs = design[:, 0] * 1e3
grating_ps = design[:, 1]
plt.plot(grating_zs, grating_ps)
plt.show()

np.savetxt('h:\\ppln_wg_apod.dat', np.vstack((grating_zs, grating_ps)).T)