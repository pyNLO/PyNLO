# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:50:51 2015
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
import exceptions

DEBUG = False

# Implemtentation of Controller struct from NR:
class Controller:
    hnext = 0.0
    errold = 0.0
    reject = False
    def __init__(self):
        self.reject = False
        self.errold = 1.0e-4
        self.hnext  = 0.0
    def success(self, err, h):
       beta=0.0
       alpha=1.0/8.0-beta*0.2
       safe=0.9
       minscale=0.333
       maxscale=6.0
       if np.isnan(h):
           raise exceptions.AssertionError('stepsize is NaN')
       if err <= 1.0:
           if err == 0.0:
               scale = maxscale
           else:
               scale = safe*np.power(err, -alpha)*np.power(err, beta)
               if scale < minscale:
                   scale = minscale
               if scale > maxscale:
                   scale = maxscale
           if self.reject:
               self.hnext = h*min(scale, 1.0)
           else:
               self.hnext = h*scale
           self.errold = max(err, 1.0e-4)
           self.reject = False           
           if DEBUG:
               print 'Accept, ',h
           return (True, h)
       else:           

           scale = max(safe*np.power(err, -alpha), minscale)
           h *= scale
           if DEBUG:
               print 'Reject, ',h           
           self.reject = True
           return (False, h)