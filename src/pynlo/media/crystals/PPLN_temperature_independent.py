# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:23:17 2015

Sellemeier coefficients and nonlinear parameter for lithium niobate
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
import numpy as np

def load(crystal_instance, params = None):
    crystal_instance.mode = 'PP'
    crystal_instance.sellmeier_type = 'a_to_f'
    crystal_instance.a    = 2.9804
    crystal_instance.b    = 0.02047
    crystal_instance.c    = 0.5981
    crystal_instance.d    = 0.0666
    crystal_instance.e    = 8.9543
    crystal_instance.f    = 416.08
    crystal_instance.deff = 14.9e-12 # from SNLO
    crystal_instance.n2   = 3e-15 / 100**2 # from Nikogosyan
    crystal_instance.pp   = lambda(x): 30.49e-6
    
    def get_index(cls, wl):
        wl_um = wl * 1e6
        return np.sqrt(1 + cls.a * wl_um**2 / (wl_um**2 - cls.b) +
                               cls.c * wl_um**2 / (wl_um**2 - cls.d) +
                               cls.e * wl_um**2 / (wl_um**2 - cls.f))
    crystal_instance.get_n = get_index