# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 12:08:09 2014
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
@author: Gabe-Local
"""
from numpy import fft

def FFT_t(A,ax=0):
    return fft.ifftshift(fft.ifft(fft.fftshift(A,axes=(ax,)),axis=ax),axes=(ax,))
def IFFT_t(A,ax=0):
    return fft.ifftshift(fft.fft(fft.fftshift(A,axes=(ax,)),axis=ax),axes=(ax,)) 

# these last two are defined in laserFOAM but never used
def FFT_x(self,A):
        return fft.ifftshift(fft.fft(fft.fftshift(A)))
def IFFT_x(self,A):
        return fft.ifftshift(fft.ifft(fft.fftshift(A)))
