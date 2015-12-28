# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:41:04 2014
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
@author: dim1
"""
import numpy as np
from scipy.interpolate import interp1d

def getGainsForGNLSE(freqs,fiber):
    baseDir = 'O:\\OFM\\Maser\\PM for OPO and DFG\\'
    fiberRoot = str(baseDir)
    if fiber == 'er7pm':
        fiberDataFile = 'er7pm_gain.csv'
    fiberData = np.genfromtxt(str(fiberRoot+fiberDataFile),delimiter=',')
    freqAxis = fiberData[0,:]
    gainInterp = interp1d(freqAxis[::-1], fiberData[1,::-1], kind='linear',bounds_error=False,fill_value=0)    
    fractGain = gainInterp(freqs)
    
    return fractGain