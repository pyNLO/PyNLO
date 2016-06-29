# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 13:53:39 2015
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

import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift
from pynlo.interactions.FourWaveMixing import global_variables
from pynlo.light.PulseBase import Pulse
import gc
import scipy.fftpack


try:
    import pyfftw
    PYFFTW_AVAILABLE=True
except:
    PYFFTW_AVAILABLE=False
    

class SSFM:
    METHOD_SSFM,METHOD_RK4IP = range(2)    
    def __init__(self,  local_error = 0.001, dz = 1e-5,
                 disable_Raman = False, disable_self_steepening = False,
                 suppress_iteration = True, USE_SIMPLE_RAMAN = False,
                 f_R = 0.18, f_R0 = 0.18, tau_1 = 0.0122, tau_2 = 0.0320):
        """
        This initialization function sets up the parameters of the SSFM.
        """
        self.iter = 0
        self.last_h = -1.0
        self.last_dir = 0.0
        self.eta = 5
        self.local_error = local_error
        self.method = SSFM.METHOD_RK4IP
        self.disable_Raman = disable_Raman
        self.disable_self_steepening = disable_self_steepening
        self.USE_SIMPLE_RAMAN = USE_SIMPLE_RAMAN

        # Raman fraction; may change depending upon which calculation method 
        # is used 
        self.f_R = f_R
        # The value for the old-style Raman response
        self.f_R0 = f_R0
        
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.dz = dz
        self.dz_min = 1e-12
        self.suppress_iteration = suppress_iteration


    def load_fiber_parameters(self, pulse_in, fiber, output_power, z=0):
        """
        This funciton loads the fiber parameters into class variables.
        """
        self.betas[:]  =  fiber.get_betas(pulse_in, z=z)
        # self.alpha[:]  = -fiber.get_gain(pulse_in, output_power) # currently alpha cannot change with z
        self.gamma     =  fiber.get_gamma(z=z)
        
        self.betas[:]  = self.conditional_fftshift(self.betas)
        # self.alpha[:]   = self.conditional_fftshift(self.alpha)
        


    def setup_fftw(self, pulse_in, fiber, output_power, raman_plots = False):
        ''' Call immediately before starting Propagate. This function does two
        things:\n
        1) it sets up byte aligned arrays for fftw\n
        2) it fftshifts betas, omegas, and the Raman response so that no further\n
            shifts are required during integration. This saves lots of time.'''

        
        self.n = pulse_in.NPTS
        
        if PYFFTW_AVAILABLE:
            
            self.fft_input    = pyfftw.empty_aligned(self.n, dtype='complex128')      
            self.fft_output   = pyfftw.empty_aligned(self.n, dtype='complex128')      
            self.ifft_input   = pyfftw.empty_aligned(self.n, dtype='complex128')      
            self.ifft_output  = pyfftw.empty_aligned(self.n, dtype='complex128')      
    
            self.fft_input_2  = pyfftw.empty_aligned(self.n, dtype='complex128')      
            self.fft_output_2 = pyfftw.empty_aligned(self.n, dtype='complex128')      
            self.ifft_input_2 = pyfftw.empty_aligned(self.n, dtype='complex128')      
            self.ifft_output_2= pyfftw.empty_aligned(self.n, dtype='complex128')      

            self.ifft_input_3 = pyfftw.empty_aligned(self.n, dtype='complex128')      
            self.ifft_output_3= pyfftw.empty_aligned(self.n, dtype='complex128')      
    
    
            # To be double sure that there are no problems, also make 2 copies of
            # the FFT objects. This lets us nest ifft_2 around a function using ifft
            # without worrying about potential problems.
            #self.fft    = pyfftw.builders.fft(self.fft_input)
            #self.fft_2  = pyfftw.builders.fft(self.fft_input_2)
            #self.ifft   = pyfftw.builders.fft(self.ifft_input)
            #self.ifft_2 = pyfftw.builders.fft(self.ifft_input_2)
            self.fft = pyfftw.FFTW(self.fft_input,
                                   self.fft_output,
                                   direction='FFTW_BACKWARD')
            self.fft_2 = pyfftw.FFTW(self.fft_input_2,
                                     self.fft_output_2,
                                     direction='FFTW_BACKWARD')
            
            self.ifft = pyfftw.FFTW(self.ifft_input,
                                    self.ifft_output,
                                    direction='FFTW_FORWARD')
            self.ifft_2 = pyfftw.FFTW(self.ifft_input_2,
                                      self.ifft_output_2,
                                      direction='FFTW_FORWARD')
            self.ifft_3 = pyfftw.FFTW(self.ifft_input_3,
                                      self.ifft_output_3,
                                      direction='FFTW_FORWARD')                                      
            
        else:
            self.fft_input    = np.ndarray((self.n,), dtype='complex128')
            self.fft_output   = np.ndarray((self.n,), dtype='complex128')
            self.ifft_input   = np.ndarray((self.n,), dtype='complex128')
            self.ifft_output  = np.ndarray((self.n,), dtype='complex128')

            self.fft_input_2  = np.ndarray((self.n,), dtype='complex128')
            self.fft_output_2 = np.ndarray((self.n,), dtype='complex128')
            self.ifft_input_2 = np.ndarray((self.n,), dtype='complex128')
            self.ifft_output_2= np.ndarray((self.n,), dtype='complex128')
                                 
            # self.fft_input    = pyfftw.empty_aligned(self.n, fft_n, dtype='complex128')
            # self.fft_output   = pyfftw.empty_aligned(self.n, fft_n, dtype='complex128')
            # self.ifft_input   = pyfftw.empty_aligned(self.n, fft_n, dtype='complex128')
            # self.ifft_output  = pyfftw.empty_aligned(self.n, fft_n, dtype='complex128')
            #
            # self.fft_input_2  = pyfftw.empty_aligned(self.n, fft_n, dtype='complex128')
            # self.fft_output_2 = pyfftw.empty_aligned(self.n, fft_n, dtype='complex128')
            # self.ifft_input_2 = pyfftw.empty_aligned(self.n, fft_n, dtype='complex128')
            # self.ifft_output_2= pyfftw.empty_aligned(self.n, fft_n, dtype='complex128')
    
            
        self.A_I    = np.ndarray((self.n,), dtype='complex128')      
        
        self.A2     = np.ndarray((self.n,), dtype='complex128')
        self.exp_D  = np.ndarray((self.n,), dtype='complex128')      
        self.k1     = np.ndarray((self.n,), dtype='complex128')      
        self.k2     = np.ndarray((self.n,), dtype='complex128')      
        self.k3     = np.ndarray((self.n,), dtype='complex128')      
        self.k4     = np.ndarray((self.n,), dtype='complex128')      
        self.temp   = np.ndarray((self.n,), dtype='complex128')      
        self.Aw     = np.ndarray((self.n,), dtype='complex128')      
        self.A2w    = np.ndarray((self.n,), dtype='complex128')      
        self.dA     = np.ndarray((self.n,), dtype='complex128')      
        self.dA2    = np.ndarray((self.n,), dtype='complex128')      
        self.R_A2   = np.ndarray((self.n,), dtype='complex128')      
        self.dR_A2  = np.ndarray((self.n,), dtype='complex128')      
        self.omegas = np.ndarray((self.n,), dtype='complex128')              
        self.alpha  = np.ndarray((self.n,), dtype='complex128')      
        self.betas  = np.ndarray((self.n,), dtype='complex128')      
        self.LinearStep_output    = np.ndarray((self.n,), dtype='complex128')              
        self.A      = np.ndarray((self.n,), dtype='complex128')              
        self.R      = np.ndarray((self.n,), dtype='complex128')              
        self.R0     = np.ndarray((self.n,), dtype='complex128')              
        self.Af     = np.ndarray((self.n,), dtype='complex128')      
        self.Ac     = np.ndarray((self.n,), dtype='complex128')      

        self.A_I[:] = 0.0        
        self.A2[:]  = 0.0
        self.Af[:]  = 0.0
        self.Ac[:]  = 0.0
        self.A[:]   = 0.0
        self.R[:]   = 0.0
        self.R0[:]  = 0.0
        
        self.omegas[:] =  pulse_in.V_THz
        self.alpha[:]  = -fiber.get_gain(pulse_in, output_power) 
        self.gamma     =  fiber.gamma
        self.w0        =  pulse_in.center_frequency_THz * 2.0 * np.pi

        self.last_h    =  None      

        # if not self.disable_Raman:
        
        self.CalculateRamanResponseFT(pulse_in)

        if raman_plots:
            plt.subplot(221)
            plt.plot(self.omegas/(2*np.pi), np.abs(self.R - (1-self.f_R)),'bo')
            plt.plot(self.omegas/(2*np.pi), np.abs(self.R0 - (1-self.f_R0)),'r')
            #plt.xlim([0,25])
            plt.title('Abs[R(w)]')
            plt.xlabel('THz')
            plt.subplot(222)
            plt.plot(self.omegas/(2*np.pi), np.unwrap(np.angle(self.R-(1-self.f_R))),'bo')
            plt.plot(self.omegas/(2*np.pi), np.unwrap(np.angle(self.R0-(1-self.f_R0))),'r')
            plt.title('Angle[R(w)]')
            plt.xlabel('THz')
            plt.subplot(223)
            plt.plot(pulse_in.T*1000, ifftshift(np.real(self.IFFT_t(\
                    self.R - (1-self.f_R)))), 'bo')
            plt.plot(pulse_in.T*1000, ifftshift(np.real(self.IFFT_t(\
                    self.R0 - (1-self.f_R0)))), 'r')
            plt.title('Abs[R[t]]')
            plt.xlim([0,1000])
            plt.xlabel('fs')
            plt.subplot(224)
            plt.plot(self.omegas/(2*np.pi), abs(self.FFT_t(self.A)))
            plt.title('Abs[A[w]]')
            plt.xlabel('THz')
            plt.show()
                
        # Load up parameters
        self.A[:]       = self.conditional_fftshift(pulse_in.AT)
        
        self.omegas[:]  = self.conditional_fftshift(self.omegas)
        # self.betas[:]   = self.conditional_fftshift(self.betas)
        self.alpha[:]   = self.conditional_fftshift(self.alpha)
        self.R[:]       = self.conditional_fftshift(self.R)
        self.R0[:]      = self.conditional_fftshift(self.R0)
        print 'pulse energy in ',np.sum(abs(pulse_in.AT))
        print 'copied as  ',np.sum(abs(self.A))
        
    #-----------------------------------------------------------------------
    # Calculates the Fourier Transform of R(T). See pg 49 of G. P. Agrawal's 
    # "Nonlinear fiber optics"  for details 
    #-----------------------------------------------------------------------
    def CalculateRamanResponseFT(self, pulse):
        """
        Calculate Raman response in frequency domain. Two versions are
        available: the first is the LaserFOAM one, which directly calculates
        R[w]. The second is Dudley-style, which calculates R[t] and then
        FFTs. Note that the use of fftshifts is critical here (FFT_t_shift)
        as is the factor of pulse_width.
        """
        
        if self.disable_Raman:
            self.R[:]=1
            return

        # Laserfoam raman function.
        TAU1 = self.tau_1
        TAU2 = self.tau_2
        F_R = self.f_R
        C = (TAU1**2+TAU2**2)/(TAU1*TAU2**2)        
        for i in xrange(pulse.NPTS):
            omega = self.omegas[i]
            H_R = C*TAU1*TAU2**2 / \
                  (TAU1**2 + TAU2**2 - 2j*omega*TAU1**2*TAU2 - TAU1**2*TAU2**2*omega**2)
            self.R0[i] = (1.0-F_R) + (F_R * H_R)
        
        # More conventional way of generating this, via Dudley    
        tau1 = self.tau_1
        tau2 = self.tau_2
        T = pulse.T_ps
        RT     = np.zeros(pulse.NPTS, dtype = 'complex128')
        if self.USE_SIMPLE_RAMAN:
            RT     = (tau1**2 + tau2**2) /( tau1 * tau2**2 )*\
                        np.exp(-T/tau2)*np.sin(T/tau1)
            RT[0:pulse.NPTS>>1]=0
            RT[:]     = RT / np.trapz(RT, T)
            #H_R    = pulse.dT*pulse.n*self.FFT_t(fftshift(RT))
            self.R[:]    = ((1.0-F_R) + pulse.time_window_ps*self.FFT_t_shift(F_R * RT))

        else:
        # Updated scheme from Lin & Agarwal 2006
            taub = 0.096
            fa = 0.75
            fb = 0.21
            fc = 0.04
            F_R = 0.245
            self.f_R = np.copy(F_R)
            ha = tau1 / (tau1**2 + tau2**2) * np.exp(-T / tau2) * np.sin(T / tau1)
            hb = (2*taub - T) / taub**2 * np.exp(-T / taub)
            RT = (fa + fc)*ha + fb*hb
            RT[0:pulse.NPTS>>1]=0
            RT[:] = RT / np.trapz(RT, T)
            self.R[:]    = ((1.0-F_R) + pulse.time_window_ps*self.FFT_t_shift(F_R * RT))               
        # R(t) = (1-fr) Delta[t] + fr ( (fa+fc)*ha(t) + fb hb(t))        
        if global_variables.USE_FREQUENCY_DOMAIN_RAMAN:
            self.R[:] = self.R0

    #-----------------------------------------------------------------------
    # Advances the current position by delta_z using an adaptive spatial
    # step algorithm.
    # See O.V. Sinkin et al, J. Lightwave Tech. 21, 61 (2003)
    # dir: 1 - Forward propagation
    #     -1 - Inverse propagation
    #-----------------------------------------------------------------------
    def integrate_over_dz(self,delta_z, direction=1):        
#        print "Propagate: delta_z",delta_z
        dist = delta_z
        dz = self.dz        

        self.last_h = -1.0  #Force an update of exp_D
        force_last_dz = False
        factor = 2**(1.0/self.eta)

        if (2.0*dz > dist):
            dz = dist/2.0

        while dist>0.0:
            self.Ac[:] = self.A
            self.Af[:] = self.A
            
            self.Ac[:] = self.Advance(self.Ac,2.0*dz,direction)
            self.Af[:] = self.Advance(self.Af,dz,direction)
            self.Af[:] = self.Advance(self.Af,dz,direction)
            #delta = |Af - Ac| / |Af| 
            delta = self.CalculateLocalError()

            old_dz = dz
            new_dz = dz
            if not self.suppress_iteration:
                print "iteration:",self.iter,"dz:",dz,"distance:", dist,\
                      "local error", delta                

            if delta > 2.0*self.local_error:
                # Discard the solution, decrease step
                new_dz = dz/2.0
                if new_dz >= self.dz_min:
                    dz = new_dz
                    # discard current step
                    continue
                else:
                    # accept step after all
                    pass
            elif (delta >= self.local_error) and (delta<=2.0*self.local_error): 
                # Keep solution, decrease step
                new_dz = dz / factor
                if new_dz >= self.dz_min:
                    dz = new_dz
                else:
                    pass
#                    printf("[%d] limited a step decrease, h = %g, delta = %g\n",self.iter,dz,delta)
            elif (delta >= (0.5*self.local_error)) and (delta<=self.local_error):
                # keep the step
                new_dz = new_dz
            else:     # delta < local_error/2
                # Step too small
                new_dz = dz * factor
                dz = new_dz
            if self.eta==3:
                self.A[:] = (4.0/3.0) * self.Af -(1.0/3.0) * self.Ac
            elif self.eta==5:
                self.A[:] = (16.0/15.0) * self.Af -(1.0/15.0) * self.Ac
            else:
                p = 2.0**(self.eta-1.0)
                self.A[:] = (p/(p-1.0)) * self.Af -(1.0/(p-1.0)) * self.Ac

            dist -= 2.0*old_dz
            self.iter += 1
#            printf("-> [%d] dz = %g dist = %g (old_dz = %g) (z = %g)\n",n,dz,dist,old_dz,self.z)
            if (2.0*dz > dist) and (dist>2.0*self.dz_min):
                force_last_dz = True
                return_dz = dz
                dz = dist/2.0
#                printf("[%d] dz = %f\n",self.iter,dz)

        if force_last_dz:
            dz = return_dz
        self.dz = dz
 
    def Advance(self,A,dz,direction):
        if self.method == SSFM.METHOD_SSFM:
            if direction==1:
                A[:] = self.LinearStep(A,dz,direction)
                return np.exp(dz*direction*self.NonlinearOperator(A))*A
            else:
                A[:] = np.exp(dz*direction*self.NonlinearOperator(A))*A
                return self.LinearStep(A,dz,direction)
        elif self.method == SSFM.METHOD_RK4IP:
            return self.RK4IP(A,dz,direction)

    def LinearStep(self,A,h,direction):
        if h!=self.last_h or direction!=self.last_dir or self.last_h is None:
            self.Calculate_expD(h,direction)
            self.last_h = h
            self.last_dir = direction           
        self.LinearStep_output[:] = self.IFFT_t(self.exp_D * self.FFT_t(A))
        return self.LinearStep_output

    def Deriv(self,Aw):
        """Calculate the temporal derivative using FFT. \n\n MODIFIED from 
        LaserFOAM original code, now input is in frequency space, output is 
        temporal derivative. This should save a few FFTs per iteration."""
        return self.IFFT_t(-1.0j*self.omegas * Aw)

    def NonlinearOperator(self,A):
        # DH commented this out on 2016-03-18
        # it should be taken care of by setting
        # self.R[:]=0 in CalculateRamanResponseFT
        # if self.disable_Raman:
 #            if self.disable_self_steepening:
 #                return 1j*self.gamma*np.abs(A)**2
 #
 #            self.Aw[:] = self.FFT_t(A)
 #            self.dA[:] = self.Deriv(A)
 #
 #            return 1j*self.gamma*np.abs(A)**2 - \
 #                   (self.gamma/self.w0)*(2.0*self.dA*A.conj() + A*self.dA.conj())
 #
 #
 #        else:
        self.A2[:]  = np.abs(A)**2   
        self.A2w[:] = self.FFT_t(self.A2)
   
        if self.disable_self_steepening:
            return 1j*self.gamma*self.IFFT_t(self.R*self.A2w)
        else:
            self.Aw[:]      = self.FFT_t(A)
            self.R_A2[:]    = self.IFFT_t(self.R*self.A2w)
            self.dA[:]      = self.Deriv(self.Aw)
            self.dA2[:]     = self.Deriv(self.A2w)
            self.dR_A2[:]   = self.IFFT_t(self.R*self.FFT_t(self.dA2))
        
            return 1j*self.gamma*self.R_A2 - (self.gamma/self.w0)* \
                   (self.dR_A2 + np.where(np.abs(A)>1.0E-15,self.dA*self.R_A2/(1.0e-20+A),0.0))


    def RK4IP(self,A,h,direction):
        """Fourth-order Runge-Kutta in the interaction picture.
           J. Hult, J. Lightwave Tech. 25, 3770 (2007)."""                   
        self.A_I[:] = self.LinearStep(A,h,direction)  #Side effect: Rely on LinearStep to recalculate self.exp_D for h/2 and direction dir                
        self.k1[:] = self.IFFT_t_2(self.exp_D*self.FFT_t_2(h*direction*self.NonlinearOperator(A)*A))
        self.k2[:] = h * direction * self.NonlinearOperator(self.A_I + self.k1/2.0)*\
                        (self.A_I + self.k1/2.0)
        self.k3[:] = h * direction * self.NonlinearOperator(self.A_I + self.k2/2.0)*\
                        (self.A_I + self.k2/2.0)        
        self.temp[:] = self.IFFT_t_2(self.exp_D*self.FFT_t_2(self.A_I+self.k3))
        self.k4[:] = h * direction * self.NonlinearOperator(self.temp)*self.temp
        if not self.suppress_iteration:
            print "ks: ",np.sum(np.abs(self.k1)),np.sum(np.abs(self.k2)),\
                    np.sum(np.abs(self.k3)),np.sum(np.abs(self.k2))
        return self.IFFT_t_2(self.exp_D * self.FFT_t_2(self.A_I + self.k1/6.0 +\
                self.k2/3.0 + self.k3/3.0)) + self.k4/6.0

    def Calculate_expD(self,h,direction):        
        self.exp_D[:] = np.exp(direction*h*0.5*(1j*self.betas-self.alpha/2.0))

    def propagate(self, pulse_in, fiber, n_steps, output_power=None, reload_fiber_each_step=False):
        """
        This is the main user-facing function that allows a pulse to be 
        propagated along a fiber (or other nonlinear medium). 
        
        Parameters
        ----------
        
        pulse_in : pulse object
            this is an instance of the :class:`pynlo.light.PulseBase.Pulse` class.
        
        fiber : fiber object
            this is an instance of the :class:`pynlo.media.fibers.fiber.FiberInstance` class.
        
        n_steps : int
            the number of steps requested in the integrator output. Note: the RK4IP integrator
            uses an adaptive step size. It should pick the correct step size automatically,
            so setting n_steps should not affect the accuracy, just the number of points that
            are returned by this funciton.
        
        output_power : 
            This parameter is a mystery
    
        reload_fiber_each_step : boolean
            This flag determines if the fiber parameters should be reloaded every step. It is 
            necessary if the fiber dispersion or gamma changes along the fiber length. 
            :func:`pynlo.media.fibers.fiber.FiberInstance.set_dispersion_function` and 
            :func:`pynlo.media.fibers.fiber.FiberInstance.set_dispersion_function` should be used
            to specify how the dispersion and gamma change with the fiber length
        
        
        Returns
        -------
        z_positions : array of float
            an array of z-positions along the fiber (in meters)
        
        AW : 2D array of complex128
            A 2D numpy array corresponding to the intensities in each *frequency* bin for each
            step in the z-direction of the fiber. 
        
        AT : 2D array of complex128
            A 2D numpy array corresponding to the intensities in each *time* bin for each
            step in the z-direction of the fiber. 
        
        pulse_out : PulseBase object
            the pulse after it has propagated through the fiber. This object is suitable for propagation 
            through the next fiber!
        """

        n_steps = int(n_steps)
        
        # Copy parameters from pulse and fiber into class-wide variables                         
        z_positions = np.linspace(0, fiber.length, n_steps + 1)
        if n_steps == 1:
            delta_z = fiber.length
        else:
            delta_z = z_positions[1] - z_positions[0]

        AW = np.complex64(np.zeros((pulse_in.NPTS, n_steps)))
        AT = np.complex64(np.copy(AW))
        
        print "Pulse energy before", fiber.fibertype,":", \
              1e9 * pulse_in.calc_epp(), 'nJ'          
              
        pulse_out = Pulse()        
        pulse_out.clone_pulse(pulse_in)
        self.setup_fftw(pulse_in, fiber, output_power)
        self.load_fiber_parameters(pulse_in, fiber, output_power) 
        

        for i in range(n_steps):                        
            print "Step:", i, "Distance remaining:", fiber.length * (1 - np.float(i)/n_steps)
            
            if reload_fiber_each_step:
                self.load_fiber_parameters(pulse_in, fiber, output_power, z=i*delta_z) 
            
            self.integrate_over_dz(delta_z)            
            AW[:,i] = self.conditional_ifftshift(self.FFT_t_2(self.A))
            AT[:,i] = self.conditional_ifftshift(self.A)
            pulse_out.set_AT(self.conditional_ifftshift(self.A))
            print "Pulse energy after:", \
              1e9 * pulse_out.calc_epp(), 'nJ'
        pulse_out.set_AT(self.conditional_ifftshift(self.A))

        print "Pulse energy after", fiber.fibertype,":", \
              1e9 * pulse_out.calc_epp(), 'nJ'
#        print "alpha out:",self.alpha
        self.cleanup()
        return z_positions, AW, AT, pulse_out
    
    def calculate_coherence(self, pulse_in, fiber, 
                            num_trials=5, random_seed=None, noise_type='one_photon_freq',
                            n_steps=50, output_power=None, reload_fiber_each_step=False):
        """
        This function runs :func:`pynlo.interactions.FourWaveMixing.SSFM.propagate` several times (given by num_trials),
        each time adding random noise to the pulse. By comparing the electric fields of the different pulses,
        and estimate of the coherence can be made. 
                            
        The parameters are the same as for :func:`pynlo.interactions.FourWaveMixing.SSFM.propagate`, except as listed below
        
        Parameters
        ----------
        
        num_trials : int
            this determines the number of trials to be run. 
        
        random_seed : int
            this is the seed for the random noise generation. Default is None, which does not set a seed for the random
            number generator, which means that the numbers will be completely randomized. 
            Setting the seed to a number (i.e., random_seed=0) will still generate random numbers for each trial,
            but the results from calculate_coherence will be completely repeatable.
        
        noise_type : str
            this specifies the method for including random noise onto the pulse. 
            see :func:`pynlo.light.PulseBase.Pulse.add_noise` for the different methods.
        
        Returns
        -------
        g12W : 2D numpy array
            This 2D array gives the g12 parameter as a function of propagation distance and the frequency.
            g12 gives a measure of the coherence of the pulse by comparing several different trials.
        
        results : list of results for each trial
            This is a list, where each item of the list contains (z_positions, AW, AT, pulse_out), the results
            obtained from :func:`pynlo.interactions.FourWaveMixing.SSFM.propagate`.
        """
        
        results = []
        for num in range(0, num_trials):

            pulse = pulse_in.create_cloned_pulse()
            pulse.add_noise(noise_type='one_photon_freq')

            y, AW, AT, pulse_out = self.propagate(pulse_in=pulse, fiber=fiber, n_steps=n_steps)

            results.append((y, AW, AT, pulse_out))

        
        for n1, (y, E1, AT, pulseout) in enumerate(results):
            for n2, (y, E2, AT, pulseout) in enumerate(results):
                if n1 == n2: continue # don't compare the same trial

                g12 = np.conj(E1)*E2/np.sqrt(np.abs(E1)**2 * np.abs(E2)**2)
                if 'g12_stack' not in locals():
                    g12_stack = g12
                else:
                    g12_stack = np.dstack((g12, g12_stack))


        # print g12_stack.shape, g12_stack.transpose().shape
        g12W = np.abs(np.mean(g12_stack, axis=2))  
        
        return g12W, results   
                            
        
        
    def propagate_to_gain_goal(self, pulse_in, fiber, n_steps, power_goal = 1,
                              scalefactor_guess = None, powertol = 0.05):
        """
        Integrate over length of gain fiber such that the average output
        power is power_goal [W]. For this to work, fiber must have spectroscopic
        gain data from an amplifier model or measurement. If the approximate
        scalefactor needed to adjust the gain is known it can be passed as
        scalefactor_guess.\n This function returns a tuple of tuples:\n
        ((ys,AWs,ATs,pulse_out), scale_factor)
        """   
        if scalefactor_guess is not None:
                scalefactor = scalefactor_guess
        else:
                scalefactor = 1
        y, AW, AT, pulse_out = self.propagate(pulse_in,
                                                 fiber,
                                                 1,
                                                 output_power = scalefactor *\
                                                                 power_goal)         
        scalefactor_revised = (power_goal / (pulse_out.calc_epp()*pulse_out.frep))
        modeled_avg_power   = pulse_out.calc_epp()*pulse_out.frep                                                                 
        output_scale = scalefactor
        while abs(modeled_avg_power - power_goal) / power_goal > powertol:
            y, AW, AT, pulse_out = self.propagate(pulse_in, fiber,
                                                     1, output_power = 
                                                     power_goal * scalefactor_revised)
            modeled_avg_power_prev = modeled_avg_power
            modeled_avg_power = pulse_out.calc_epp()*pulse_out.frep
            slope = (modeled_avg_power - modeled_avg_power_prev) /\
                            (scalefactor_revised - scalefactor)                            
            yint = modeled_avg_power_prev - slope*scalefactor    
            # Before updating the scale factor, see if the new or old modeled
            # power is closer to the goal. When the loop ends, whichever is more
            # accurate is used for final iteration. This makes maximum use of 
            # each (computationally expensive) numeric integration
            if abs(modeled_avg_power-power_goal) < \
                        abs(modeled_avg_power_prev-power_goal):
                output_scale = scalefactor_revised
            else:
                output_scale = scalefactor
            # Update scalefactor and go to top of loop
            scalefactor = scalefactor_revised
            scalefactor_revised = (power_goal - yint)/slope                            
        print 'Updated final power:',modeled_avg_power,'W'        
        return (self.propagate(pulse_in, fiber,
                              n_steps,
                              output_power = power_goal * output_scale),
                output_scale)
    def test_raman(self, pulse_in, fiber, output_power = 0): 
        ''' Function for testing Raman response function. This plots Generates 
            R[w] and makes plots, but does not actually integrate anything.'''
        # Copy parameters from pulse and fiber into class-wide variables                    
        print "Pulse energy before", fiber.fibertype,":", \
              1e9 * pulse_in.calc_epp(), 'nJ'          
        self.setup_fftw(pulse_in, fiber, output_power, raman_plots = True)

    ### Lots of boring FFT code from here on out.
    def FFT_t(self, A):
        if PYFFTW_AVAILABLE:
            if global_variables.PRE_FFTSHIFT:
                self.fft_input[:] = A
                return self.fft()                
            else:
                self.fft_input[:] = fftshift(A)
                return ifftshift(self.fft())
        else:
            if global_variables.PRE_FFTSHIFT:
                return scipy.fftpack.ifft(A)
            else:
                return ifftshift(scipy.fftpack.ifft(fftshift(A)))
    def IFFT_t(self, A):        
        if PYFFTW_AVAILABLE:
            if global_variables.PRE_FFTSHIFT:
                self.ifft_input[:] = A
                return self.ifft()
            else:
                self.ifft_input[:] = fftshift(A)
                return ifftshift(self.ifft())
        else:
            if global_variables.PRE_FFTSHIFT:
                return scipy.fftpack.fft(A)
            else:
                return ifftshift(scipy.fftpack.fft(fftshift(A)))
    def FFT_t_shift(self, A):
        if PYFFTW_AVAILABLE:
            self.fft_input[:] = fftshift(A)
            return ifftshift(self.fft())
        else:
            return ifftshift(scipy.fftpack.ifft(fftshift(A)))
    def IFFT_t_shift(self, A):
        if PYFFTW_AVAILABLE:
            self.ifft_input[:] = fftshift(A)
            return ifftshift(self.ifft())
        else:
            return ifftshift(scipy.fftpack.fft(fftshift(A)))
    def FFT_t_2(self, A):        
        if PYFFTW_AVAILABLE:
            if global_variables.PRE_FFTSHIFT:
                self.fft_input_2[:] = A
                return self.fft_2()
            else:
                self.fft_input_2[:] = fftshift(A)
                return ifftshift(self.fft_2())
        else:
            if global_variables.PRE_FFTSHIFT:
                return scipy.fftpack.ifft(A)
            else:
                return ifftshift(scipy.fftpack.ifft(fftshift(A)))
    def IFFT_t_2(self, A):        
        if PYFFTW_AVAILABLE:
            if global_variables.PRE_FFTSHIFT:
                self.ifft_input_2[:] = A
                return self.ifft_2()
            else:
                self.ifft_input_2[:] = fftshift(A)
                return ifftshift(self.ifft_2())
        else:
            if global_variables.PRE_FFTSHIFT:
                return scipy.fftpack.fft(A)
            else:
                return ifftshift(scipy.fftpack.fft(fftshift(A)))
    def IFFT_t_3(self, A):        
        if PYFFTW_AVAILABLE:
            if global_variables.PRE_FFTSHIFT:                
                self.ifft_input_3[:] = A
                return self.ifft_3()
            else:
                self.ifft_input_3[:] = fftshift(A)
                return ifftshift(self.ifft_3())
        else:
            if global_variables.PRE_FFTSHIFT:
                return scipy.fftpack.fft(A)
            else:
                return ifftshift(scipy.fftpack.fft(fftshift(A)))
            

    #-----------------------------------------------------------------------
    # Calculates the relative local error.
    # See O.V. Sinkin et al, J. Lightwave Tech. 21, 61 (2003)
    # Returns |Af - Ac| / |Af|
    #-----------------------------------------------------------------------
    def CalculateLocalError(self):
        denom = np.linalg.norm(self.Af)
        if denom != 0.0:
            return np.linalg.norm(self.Af-self.Ac)/np.linalg.norm(self.Af)
        else:
            return np.linalg.norm(self.Af-self.Ac)
    def conditional_ifftshift(self, x):
        if global_variables.PRE_FFTSHIFT:
            x[:] = ifftshift(x)
            return x
        else:
            return x
    def conditional_fftshift(self, x):
        if global_variables.PRE_FFTSHIFT:
            x[:] = fftshift(x)
            return x
        else:
            return x            
    def cleanup(self):
        gc.collect()
