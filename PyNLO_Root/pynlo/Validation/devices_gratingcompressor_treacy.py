# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:06:10 2015
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

from pynlo.devices.grating_compressor import TreacyCompressor
from pynlo.light.DerivedPulses import GaussianPulse
import numpy as np
from scipy import constants
from matplotlib import pyplot as plt

# Check against Tom Allison's calculation
theta = 41.5
ruling = 1250.0

print "Allison's result:  GDD = (-) 1.8 ps^2, TOD = (+) 7.793 10^6 fs^3"
print 'TOD/GDD =', 1.e15*-7.793e6*1.e-45/(1.8e-24), 'fs'

TC = TreacyCompressor(ruling, theta)

wl = 1060.0
sep = 11.5e-2

gdd = TC.calc_compressor_gdd(wl, sep)
print gdd * 1.0e24, 'ps^2'

wls  = np.linspace(1050, 1070)
ws   = 2.0*np.pi*299792458.0 / (wls*1.0e-9)
w0   = 2.0*np.pi*299792458.0 / (wl*1.0e-9)

phis = TC.calc_dphi_domega(ws, sep)
fit = np.polyfit(ws, phis, 3)
gddpoly = np.polyder(fit)
gdd_from_fit = np.poly1d(gddpoly)(w0)*2.0
print 'GDD from polyfit is ', gdd_from_fit
todpoly = np.polyder(fit, m=2)
tod_from_fit = np.poly1d(todpoly)(w0)*2.0
print 'TOD from phase polyfit is ', tod_from_fit* 1.0e45*1.0e-6, ' 10^6 fs^3'

betas = TC.calc_compressor_gdd(wls, sep)
fit = np.polyfit(ws, betas, 3)
todpoly = np.polyder(fit)
tod_from_fit = np.poly1d(todpoly)(w0)
print 'Fit of GDD derivative: ', tod_from_fit * 1.0e45*1.0e-6, ' 10^6 fs^3'


tod = TC.calc_compressor_HOD(wl, sep, 3)
print tod * 1.0e36, 'ps^3'
print tod * 1.0e45*1.0e-6, ' 10^6 fs^3'
print 'numerically calculated tod/gdd = ', tod/gdd*1.0e15


gdd_deriv = TC.calc_compressor_dnphi_domega_n(wl, sep, 1)
print gdd_deriv*-2.0

p = GaussianPulse(0.100, 0.1, 1060., time_window = 10.0, power_is_avg = True)
plt.plot(p.T_ps, abs(p.AT)**2)
p.chirp_pulse_W(1.80262920634, -0.00780201762947)
plt.plot(p.T_ps, abs(p.AT)**2)
TC.apply_phase_to_pulse(sep, p)
plt.plot(p.T_ps, abs(p.AT)**2,'ok')
plt.show()