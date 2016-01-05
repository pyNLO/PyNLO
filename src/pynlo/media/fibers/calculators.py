# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:56:17 2014
This file is part of pyNLO.

    pyNLO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public gLicense as published by
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
from scipy.misc import factorial
from scipy import constants
import matplotlib.pyplot as plt

def DTabulationToBetas(lambda0, DData, polyOrder, DDataIsFile = True, return_diagnostics = False):
    """ Read in a tabulation of D vs Lambda. Returns betas in array 
    [beta2, beta3, ...]. If return_diagnostics is True, then return
    (betas, fit_x_axis (omega in THz), data (ps^2), fit (ps^2) ) """
    # 
    # Expand about lambda0
    makePlots = 0
    if DDataIsFile:
        DTab = np.genfromtxt(DData,delimiter=',',skiprows=1)
    else:
        DTab = DData[:]
            
    # Units of D are ps/nm/km
    # Convert to s/m/m 
    DTab[:,1] = DTab[:,1] * 1e-12 * 1e9 * 1e-3
    c = constants.speed_of_light
    
    omegaAxis = 2*np.pi*c  / (DTab[:,0]*1e-9) - 2*np.pi*c  /(lambda0 * 1e-9)
    # Convert from D to beta via  beta2 = -D * lambda^2 / (2*pi*c) 
    
    betaTwo = -DTab[:,1] * (DTab[:,0]*1e-9)**2 / (2*np.pi*c) 
    # The units of beta2 for the GNLSE solver are ps^2/m; convert
    betaTwo = betaTwo * 1e24
    # Also convert angular frequency to rad/ps
    omegaAxis = omegaAxis * 1e-12 #  s/ps
    
    # How betas are interpreted in gnlse.m:
    #B=0;
    #for i=1:length(betas)
    #    B = B + betas(i)/factorial(i+1).*V.^(i+1);
    #end
    
    # Fit beta2 with high-order polynomial
    polyFitCo = np.polyfit(omegaAxis, betaTwo, polyOrder)
    
    Betas = polyFitCo[::-1]
    
    polyFit = np.zeros((len(omegaAxis),))   

    for i in range(len(Betas)):
        Betas[i] = Betas[i] * factorial(i)
        polyFit = polyFit + Betas[i] / factorial(i)*omegaAxis**i
    
    if makePlots == 1:
#        try:
#            set(0,'CurrentFigure',dispfig);
#        catch ME
#            dispfig = figure('WindowStyle', 'docked');
#        end        
        plt.plot(omegaAxis, betaTwo,'o')
        plt.plot(omegaAxis, polyFit)
        plt.show()
    if return_diagnostics:
        return Betas, omegaAxis, betaTwo, polyFit
    else:
        return Betas