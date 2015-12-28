# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:25:25 2013
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
@author: Dan
"""

import numpy as np
import scipy.fftpack as fftpack
import pyfftw
from copy import deepcopy
from scipy.integrate import odeint
from scipy import constants
from pynlo.interactions.ThreeWaveMixing import field_classes


class NLmix: 
    fields = None
    def __init__(self, pump_in, signal_in, idler_in, crystal_in, zsteps = 1e2,
                 disable_SPM = False, waist = 10e-6):
        
        self.waist = waist        
        self.fields = field_classes.ThreeFields(NPTS)
        
        self.pump = deepcopy(pump_in)
        self.signal = deepcopy(signal_in)
        self.idler = deepcopy(idler_in)
        self.crystal = deepcopy(crystal_in)
        
        self.disable_SPM = disable_SPM 
        
        self.pump.name = 'pump'
        self.signal.name = 'signal'
        self.idler.name = 'idler'

        self.c = constants.speed_of_light
        self.eps = constants.epsilon_0
        self.frep = self.signal.frep
        self.veclength = self.signal.n

        self.pump.waist = self.waist #47e-6
        self.signal.waist = self.waist #77e-6
        self.idler.waist = self.waist #np.mean([self.waist_p, self.waist_s])
        
        self.length = crystal_in.length
        self.zsteps = zsteps
        self.z = np.linspace(0, self.length, self.zsteps)
        
        self.fftobject = fftcomputer(self.veclength) 
        
        self.pump = self.pulse_in_crystal(self.pump)
        self.signal = self.pulse_in_crystal(self.signal)
        self.idler = self.pulse_in_crystal(self.idler)
                    
        self.tmax = 1. / (2. *(self.signal.V[1] - self.signal.V[0]))
        self.time = np.linspace(-self.tmax, self.tmax, self.veclength)  
        
        
        self.ystart = np.array(np.hstack((self.pump.get_AW(),
                                      self.signal.get_AW(),
                                      self.idler.get_AW())),
                                      dtype='complex128')
        
        self.AsAi   = np.zeros((self.veclength,), dtype=np.complex128)
        self.ApAi   = np.zeros((self.veclength,), dtype=np.complex128)
        self.ApAs   = np.zeros((self.veclength,), dtype=np.complex128)

        self.phi_p   = np.zeros((self.veclength,), dtype=np.complex128)
        self.phi_s   = np.zeros((self.veclength,), dtype=np.complex128)        
        self.phi_i   = np.zeros((self.veclength,), dtype=np.complex128)        
        
        self.dApdZ   = np.zeros((self.veclength,), dtype=np.complex128)        
        self.dAsdZ   = np.zeros((self.veclength,), dtype=np.complex128)        
        self.dAidZ   = np.zeros((self.veclength,), dtype=np.complex128)        
        
        self.dApdZ = []
    
        if self.disable_SPM:
            [self.jl_p, self.jl_s, self.jl_i] = np.zeros((3, self.veclength))    
    
    def pulse_in_crystal(self, pulse):
        if self.crystal.mode == 'BPM':
            if pulse.name == 'pump':
                pulse.n  = self.crystal.nmix(pulse.wl)
                pulse.n0 = self.crystal.nmix(pulse.center_wl)         
            else:
                pulse.n  = self.crystal.no(pulse.wl)
                pulse.n0 = self.crystal.no(pulse.center_wl)
        if self.crystal.mode == 'PP':
            pulse.n  = self.crystal.n(pulse.wl)
            pulse.n0 = self.crystal.n(pulse.center_wl)
        if self.crystal.mode == 'simple':
            pulse.n  = self.crystal.n0 * np.ones(self.veclength)
            pulse.n0 = self.crystal.n0
        pulse.k0     = 2. * np.pi * pulse.n0 / pulse.center_wl
        pulse.k      = 2. * np.pi * pulse.n / pulse.wl - pulse.k0
        pulse.w      = 2. * np.pi * self.c / pulse.wl
        pulse.vg     = self.vg(pulse.n, pulse.wl)
        pulse.vg_k   = self.vg_k(pulse.w, pulse.k + pulse.k0)
        pulse.gamma  = self.gamma(pulse)
        pulse.e_to_a = np.sqrt(2 * np.pi * pulse.waist**2 * pulse.n *
                               self.eps * self.c) # conversion from electric field to A in code
        return pulse
        
    def derivOLD(self, x, y):
        delta = np.diff(y) / np.diff(x)
        return np.append(delta,delta[-1]) 
                   
    def vg(self, n, wl):
            return self.c / (n - wl * self.deriv(wl, n))
    
    def vg_k(self, w, k):
        return self.deriv(k, w)
        
    def gamma(self, pulse):
            return self.eps / 2. * pulse.n**2 * pulse.vg
     
    def gen_jl(self, y):
        
        pump   = self.Apreal(y) + 1j*self.Apimag(y)
        signal = self.Asreal(y) + 1j*self.Asimag(y)
        idler  = self.Aireal(y) + 1j*self.Aiimag(y)        
        
        jl = np.zeros((3, self.veclength), dtype = 'complex64')
        gamma = [self.pump.gamma, self.signal.gamma, self.idler.gamma]
        waist = [self.pump.waist, self.signal.waist, self.idler.waist]
        
        i = 0      
        
        for vec1 in [pump, signal, idler]:
            for vec2 in [pump, signal, idler]:
                if np.all(vec1 == vec2):
                    jl[i] = jl[i] + (1. / (2.*np.pi) * gamma[i] *
                          self.fftobject.corr(vec2, vec2) * np.sqrt(2. /
                          (self.c * self.eps * np.pi * waist[i]**2)))

                else:
                    jl[i] = jl[i] + (1. / np.pi * gamma[i] *
                          self.fftobject.corr(vec2, vec2) * np.sqrt(2. /
                          (self.c * self.eps * np.pi * waist[i]**2)))
            i += 1
        
        [self.jl_p, self.jl_s, self.jl_i] = jl  
        
        return 1
    
    def poling(self, x):
        return np.sign(np.sin(2. * np.pi * x / self.crystal.pp(x)))
    
    # Integrand:
    # State is defined by:
    # 1.) fields in crystal
    # 2.) values of k
    # 3.) electric field->intensity conversion (~ area)

    # Output is vector of estimate for d/dz field
        
    def derivXX(self, z, y, dydx):        
        if self.crystal.mode == 'PP':          
            deff = self.poling(z) * self.crystal.deff   
        if self.crystal.mode == 'BPM' or self.crystal.mode == 'simple':
            deff = self.crystal.deff
        # After epic translation of Dopri853 from Numerical Recipes' C++ to
        # native Python/NumPy, we can use complex numbers throughout:
        self.phi_p[:] = np.exp(1j * (self.pump.k + self.pump.k0) * z)
        self.phi_s[:] = np.exp(1j * (self.signal.k + self.signal.k0) * z)
        self.phi_i[:] = np.exp(1j * (self.idler.k + self.idler.k0) * z)
        
        if not self.disable_SPM:
            self.gen_jl(y)
        
        self.AsAi[:] = self.phi_p**-1 * self.fftobject.conv(self.As * self.phi_s, self.Ai * self.phi_i)
        self.ApAs[:] = self.phi_i**-1 * self.fftobject.corr(self.Ap * self.phi_p, self.As * self.phi_s)
        self.ApAi[:] = self.phi_s**-1 * self.fftobject.corr(self.Ap * self.phi_p, self.Ai * self.phi_i)
        
        # np.sqrt(2 / (c * eps * pi * waist**2)) converts to electric field        
        # If the chi-3 terms are included:
        if not self.disable_SPM:
            jpap = self.phi_p**-1 * self.fftobject.conv(self.jl_p, self.Ap * self.phi_p) * \
                   np.sqrt(2. / (constants.speed_of_light * constants.epsilon_0 * np.pi * self.waist_p**2))
            jsas = self.phi_s**-1 * self.fftobject.conv(self.jl_s,
                   self.As * self.phi_s) * \
                   np.sqrt(2. / (constants.speed_of_light* constants.epsilon_0 * np.pi * self.waist_s**2))
            jiai = self.phi_i**-1 * self.fftobject.conv(self.jl_i,
                   self.Ai * self.phi_i) * \
                   np.sqrt(2. / (constants.speed_of_light* constants.epsilon_0 * np.pi * self.waist_i**2))      
            self.dApdZ[:] = 1j * 2 * self.AsAi * self.pump.w * deff / (constants.speed_of_light* self.pump.n) * \
                    self.pump.e_to_a / self.signal.e_to_a / self.idler.e_to_a\
                    -1j * self.pump.w * self.crystal.n2 / (2.*np.pi*self.c) * jpap
            self.dAsdZ[:] = 1j * 2 * self.ApAi * self.signal.w * deff / (constants.speed_of_light* self.signal.n) * \
                    self.signal.e_to_a / self.idler.e_to_a / self.pump.e_to_a\
                    -1j * self.signal.w * self.crystal.n2 / (2.*np.pi*self.c) * jsas
            self.dAidZ[:] = 1j * 2 * self.ApAs * self.idler.w * deff / (constants.speed_of_light* self.idler.n) * \
                    self.idler.e_to_a / self.pump.e_to_a / self.signal.e_to_a\
                                    -1j * self.idler.w * self.crystal.n2 / (2.*np.pi*self.c) * jiai
        else:
            # Only chi-2:
            self.dApdZ[:] = 1j * 2 * self.AsAi * self.pump.w * deff / (constants.speed_of_light* self.pump.n) * \
                    self.pump.e_to_a / self.signal.e_to_a / self.idler.e_to_a
            self.dAsdZ[:] = 1j * 2 * self.ApAi * self.signal.w * deff / (constants.speed_of_light* self.signal.n) * \
                    self.signal.e_to_a / self.idler.e_to_a / self.pump.e_to_a
            self.dAidZ[:] = 1j * 2 * self.ApAs * self.idler.w * deff / (constants.speed_of_light* self.idler.n) * \
                    self.idler.e_to_a / self.pump.e_to_a / self.signal.e_to_a
        L = len(self.ApAi)
        dydx[0:L ] = self.dApdZ
        dydx[L:2*L] = self.dAsdZ
        dydx[2*L:3*L] = self.dAidZ

    
    def run(self):
#        solver = odespy.Dop853(self.mix) #, atol = 1e-12, rtol = 1e-10)
#        solver.set_initial_condition(self.ycurrent)
#        soln, zfinal = solver.solve(self.z)
        soln = odeint(self.mix, self.ycurrent, self.z)
        self.soln = np.transpose(soln)
        
        return self.pump_out(), self.signal_out(), self.idler_out()

    def pump_out(self):
        return (self.Apreal(self.soln) + 1j * self.Apimag(self.soln)) * \
                np.exp(1j * np.outer((self.pump.k + self.pump.k0), self.z)) 
    def signal_out(self):
        return (self.Asreal(self.soln) + 1j * self.Asimag(self.soln)) * \
                np.exp(1j * np.outer((self.signal.k + self.signal.k0), self.z))
    def idler_out(self):
        return (self.Aireal(self.soln) + 1j * self.Aiimag(self.soln)) * \
                np.exp(1j * np.outer((self.idler.k + self.idler.k0), self.z)) 
    
class fftcomputer:
    def __init__(self, gridsize):
        self.gridsize = gridsize
        self.corrin = pyfftw.n_byte_align_empty(gridsize*2,16,'complex128')
        self.corrtransfer = pyfftw.n_byte_align_empty(gridsize*2,16,'complex128')
        self.fft = pyfftw.FFTW(self.corrin,self.corrtransfer,direction='FFTW_FORWARD')
        
        self.backout = pyfftw.n_byte_align_empty(gridsize*2,16,'complex128')
        self.ifft = pyfftw.FFTW(self.corrtransfer,self.backout,direction='FFTW_BACKWARD')
        
    def corr(self, data1, data2):
        n = self.gridsize
        self.corrin[:] = 0
        self.corrin[:n] = data2
        temp = np.conjugate(np.copy(self.fft()))
        
        self.corrin[:] = 0
        self.corrin[:n] = data1
        ans = self.fft()
        ans[:] = ans*temp
        
        return fftpack.ifftshift(np.copy(self.ifft()))[(n>>1):n+(n>>1)]
        
    def conv(self, resp, sig):
        n = self.gridsize
        self.corrin[:] = 0 
        self.corrin[n:] = resp
        temp = np.copy(self.fft())
        
        self.corrin[:] = 0
        self.corrin[:n] = sig
        ans = self.fft()
        ans[:] = ans*temp
        
        return fftpack.ifftshift(np.copy(self.ifft()))[(n>>1):n+(n>>1)]
        