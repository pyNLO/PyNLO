# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08 14:08:12 2015
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
    
Containers for three wave mixing fields (pump, signal, idler)

@author: ycasg
"""

import numpy as np
import exceptions

PUMP = 0
SGNL = 1
IDLR = 2

class beam_parameters:
    """ Class for managing physical beam parameters of a three-wave mixing 
    problem. Initally, only supports fixed beam size, but could/should be
    extended to spatial overlap by physical diffraction. """
    pump_waist = None
    idlr_waist = None
    sgnl_waist = None
    waists  = []
    def __init__(self, pump, sgnl, idlr):
        self.pump_waist = pump
        self.sgnl_waist = sgnl
        self.idlr_waist = idlr
        self.waists     = [pump, sgnl, idlr]
        
    def calculate_overlap(self, A, B):
        """ calculate overlap integral between fields A and B. A & B must be
        integers between 0-2 (0 = pump, 1 = signal, 3 = idler.)"""
        if (type(A) is not int) or (type(B) is not int):
            e = exceptions.TypeError('A & B must both be integers.')
            raise e
        if A < 0 or A>3 or B<0 or B>3:
            e = exceptions.ValueError('A & B must be in range [0,3].')
            raise e
        return (self.waists[A]+self.waists[B]) / 2.0
        
        

class ThreeFields:
    """ Simple class for holding pump, signal, and idler fields. This daata is 
    a glorified 3xN array of complex128. Can easily be made a C/C++ type. """
    fields  = None
    field_derivs  = None
    ks      = None
    k0s     = [0.0, 0.0, 0.0]
    NPTS    = 0
    def __init__(self, NPTS):
        self.NPTS = NPTS
        self.create_fields()        
    def create_fields(self):
        """ Allocate arrays for field k's and complex amplitudes."""
        self.fields       = np.zeros( (3, self.NPTS), dtype = np.complex128)
        self.field_derivs = np.zeros( (3, self.NPTS), dtype = np.complex128)
        self.ks           = np.zeros( (3, self.NPTS), dtype = np.double)
        
    def set_pump(self, field):
        self.fields[PUMP,:] = field
    def set_sgnl(self, field):
        self.fields[SGNL,:] = field
    def set_idlr(self, field):
        self.fields[IDLR,:] = field

    def get_pump(self):
        return self.fields[PUMP,:]
    def get_sgnl(self):
        return self.fields[SGNL,:]
    def get_idlr(self):
        return self.fields[IDLR,:]
        
    def set_all(self, fields):
        self.fields[:] = fields
    def get_all(self):
        return self.fields
    
    def create_copy(self):
        new_obj = ThreeFields(self.NPTS)
        new_obj.set_all(self.get_all())
        return new_obj
    
