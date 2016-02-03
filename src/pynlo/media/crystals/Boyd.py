# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:23:17 2015

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
    along with pyNLO.  If not, see <http://www.gnu.org/licenses/>
@author: ycasg
"""

def load(crystal_instance, params):
    """ Load simple crystal with fixed refractive index and deff. """
    crystal_instance.mode = 'simple'
    crystal_instance.n0   = 2.
    crystal_instance.deff = 4e-12
    crystal_instance.n2   = 0

    def get_index(crystal_instance, wl):
        return crystal_instance.n0
    crystal_instance.get_n = get_index