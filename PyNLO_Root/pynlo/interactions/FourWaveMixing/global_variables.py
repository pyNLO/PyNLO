# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 13:56:18 2015
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

###     Global variables    ###
#   USE_PYFFTW : True   - > use pyfftw
#                False  - > use numpy fft
USE_PYFFTW = False
#   USE_FREQUENCY_DOMAIN_RAMAN : 
#   True   - > calculate Raman respose in frequency domain (older)
#   False  - > calculate Raman reponse in time domain (Modern version)
USE_FREQUENCY_DOMAIN_RAMAN = False
#   USE_SIMPLE_RAMAN : 
#   True   - > use classic (Agarwal 1989) sin(t/t1)exp(-t/t2) response
#   False  - > use more modern three-time version (Lin & Agarwal 2006)
PRE_FFTSHIFT = True