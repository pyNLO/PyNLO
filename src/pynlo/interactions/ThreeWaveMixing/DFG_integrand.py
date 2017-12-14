# -*- coding: utf-8 -*-
"""
Difference frequency generation module

Defines:
    - The dfg_problem, a class which can be intergrated by the pyNLO ODESolve
    - The fftcomputer, which handles FFTs using pyFFTW
    - A helper class, dfg_results_interface, which provides a Pulse-class based
      wrapper around the dfg results.     


Authors: Dan Maser, Gabe Ycas
"""



import numpy as np
import scipy.fftpack as fftpack
from copy import deepcopy
from scipy import constants
from pynlo.light import OneDBeam, OneDBeam_highV_WG
import exceptions
from pynlo.light.DerivedPulses import NoisePulse
from pynlo.light.PulseBase import Pulse
from matplotlib import pyplot as plt
import logging

try:
    import pyfftw
    PYFFTW_AVAILABLE=True
except:
    PYFFTW_AVAILABLE=False


class dfg_problem:     
    """
    This class defines the integrand for a DFG or OPO parametric inteaction.
    Following Eqn (8) in Seres & Hebling, "Nonstationary theory of synchronously pumped femtosecond optical parametric oscillators", JOSA B Vol 17 No 5, 2000. 
    
    """
    last_calc_z = -1e6
    overlap_pump = None
    overlap_sgnl = None
    overlap_idlr = None
    pump_P_to_a = None
    sgnl_P_to_a = None
    idlr_P_to_a = None
    _plot_beam_overlaps = False
    _wg_mode   = False
    _Aeff_squm      = 0.0
    _pump_center_idx = 0.0
    _prev_pp_boundary = None
    _next_pp_boundary = None
    _pp_sign = 1
    _pp_last = 0
    
    def __init__(self, pump_in, sgnl_in, crystal_in,
                 disable_SPM = False, pump_waist = 10e-6,
                 apply_gouy_phase = False, plot_beam_overlaps = False,
                 wg_mode = False, Aeff_squm = None):
        """ Initialize DFG problem. The idler field must be derived from the
        signal & idler, as its center frequency is exactly the difference-
        frequency between pump & signal.
        
        Setting the apply_gouy_phase flag to True enables the calculation of the
        wavelength-dependent Gouy phase shift. This is disabled by default because
        it is slow (if deemed important it could be sped up by inteprolation, but
        the effect of Gouy phase seems small so it might not be worthwhile.) """
        self._wg_mode = wg_mode
        
        if self._wg_mode == False:
            self.waist = pump_waist
            self._plot_beam_overlaps = plot_beam_overlaps    
            self._calc_gouy = apply_gouy_phase
        else:
            assert Aeff_squm is not None
            self._Aeff_squm = Aeff_squm
            self._plot_beam_overlaps = False
            self._calc_gouy = apply_gouy_phase                    
        

        # Create idler Pulse.
        
        # The idler grid must be centered to match DFG of the pump and signal
        # center frequencies. The center matching is implicitly used in the
        # mixing calculations to conserve energy
        idler_cwl_natural = 1.0/(1.0/pump_in.center_wavelength_nm -\
                             1.0/sgnl_in.center_wavelength_nm)   
        idlr_in = NoisePulse(center_wavelength_nm   = idler_cwl_natural, 
                             frep_MHz               = pump_in.frep_MHz,
                             NPTS                   = pump_in.NPTS,
                             time_window_ps         = pump_in.time_window_ps)        
        

        # Double check that fields do not overlap
        if ( max(pump_in.wl_nm) > min(sgnl_in.wl_nm) ): 
            raise exceptions.ValueError("Pump and signal field grids overlap.")
        if ( max(sgnl_in.wl_nm) > min(idlr_in.wl_nm) ):
            print "sgnl max: ", max(sgnl_in.wl_nm)
            print "idlr min: ", min(idlr_in.wl_nm)
            raise exceptions.ValueError("Signal and idler field grids overlap.")
        self.idlr_in = idlr_in

        self.pump = deepcopy(pump_in)
        self.sgnl = deepcopy(sgnl_in)
        self.idlr = deepcopy(idlr_in)
        
        self.crystal = deepcopy(crystal_in)
        if self.crystal.mode == 'PP':          
            self.precompute_poling()
        
        self.disable_SPM = disable_SPM 
        
        self.c = constants.speed_of_light
        self.eps = constants.epsilon_0
        self.frep = sgnl_in.frep_mks
        self.veclength = sgnl_in.NPTS
        if not pump_in.NPTS == sgnl_in.NPTS == idlr_in.NPTS:
            raise exceptions.ValueError("""Pump, signal, and idler do not have
                                            same length.""")
        if self._wg_mode == False:
            if self.crystal.mode == 'BPM':
                self.pump_beam = OneDBeam(self.waist, this_pulse = self.pump, axis = 'mix')
                self.pump_beam.set_waist_to_match_central_waist(self.pump, self.waist, self.crystal)
                # Pump beam sets all other beams' confocal parameters
                self.sgnl_beam = OneDBeam(self.waist, this_pulse = self.sgnl ,axis = 'o')    
                self.sgnl_beam.set_waist_to_match_confocal(self.sgnl, self.pump, self.pump_beam, self.crystal)
                self.idlr_beam = OneDBeam(self.waist , this_pulse = self.idlr,axis = 'o')
                self.idlr_beam.set_waist_to_match_confocal(self.idlr, self.pump, self.pump_beam, self.crystal)
            else:
                self.pump_beam = OneDBeam(waist_meters = self.waist, this_pulse = self.pump)
                self.pump_beam.set_waist_to_match_central_waist(self.pump, self.waist, self.crystal)
                self.sgnl_beam = OneDBeam(waist_meters = self.waist, this_pulse = self.sgnl)
                self.sgnl_beam.set_waist_to_match_confocal(self.sgnl, self.pump, self.pump_beam, self.crystal)
                self.idlr_beam = OneDBeam(waist_meters = self.waist, this_pulse = self.idlr)
                self.idlr_beam.set_waist_to_match_confocal(self.idlr, self.pump, self.pump_beam, self.crystal)
        else:
            # Waveguide mode. Only valid for ZZZ phase matching
            assert self.crystal.mode != 'BPM'
            self.pump_beam = OneDBeam_highV_WG(Aeff_squm = self._Aeff_squm, this_pulse = self.pump)
            self.sgnl_beam = OneDBeam_highV_WG(Aeff_squm = self._Aeff_squm, this_pulse = self.sgnl)
            self.idlr_beam = OneDBeam_highV_WG(Aeff_squm = self._Aeff_squm, this_pulse = self.idlr)
            
        self.fftobject = fftcomputer(self.veclength)                 
        self.ystart = np.array(np.hstack((self.pump.AW,
                                          self.sgnl.AW,
                                          self.idlr.AW )),
                                          dtype='complex128')

        # Preallocated mixing terms (work spaces)
        self.AsAi   = np.zeros((self.veclength,), dtype=np.complex128)
        self.ApAi   = np.zeros((self.veclength,), dtype=np.complex128)
        self.ApAs   = np.zeros((self.veclength,), dtype=np.complex128)
        # Preallocated phasors for adding linear dispersion (work spaces)
        self.phi_p   = np.zeros((self.veclength,), dtype=np.complex128)
        self.phi_s   = np.zeros((self.veclength,), dtype=np.complex128)        
        self.phi_i   = np.zeros((self.veclength,), dtype=np.complex128)        
        # Preallocated outputs (work spaces)
        self.dApdZ   = np.zeros((self.veclength,), dtype=np.complex128)        
        self.dAsdZ   = np.zeros((self.veclength,), dtype=np.complex128)        
        self.dAidZ   = np.zeros((self.veclength,), dtype=np.complex128)        
        # Relative wave numbers
        self.k_p = self.pump_beam.get_k_in_crystal(pump_in, self.crystal)
        self.k_s = self.sgnl_beam.get_k_in_crystal(sgnl_in, self.crystal)
        self.k_i = self.idlr_beam.get_k_in_crystal(idlr_in, self.crystal)

        self.k_p_0  = self.k_p[int(len(self.k_p)/2.0)]
        self.k_s_0  = self.k_s[int(len(self.k_s)/2.0)]
        self.k_i_0  = self.k_i[int(len(self.k_i)/2.0)]
        
        self.k_p    -= self.k_p_0
        self.k_s    -= self.k_s_0
        self.k_i    -= self.k_i_0        
        self.n_p  = self.pump_beam.get_n_in_crystal(self.pump, self.crystal)
        self.n_s  = self.sgnl_beam.get_n_in_crystal(self.sgnl, self.crystal)
        self.n_i  = self.idlr_beam.get_n_in_crystal(self.idlr, self.crystal)      
        
        self._pump_center_idx = np.argmax(abs(self.pump.AW))

        self.approx_pulse_speed = max([max(constants.speed_of_light / self.n_p),
                                       max(constants.speed_of_light / self.n_s),
                                       max(constants.speed_of_light / self.n_i)])
        
        if not self.disable_SPM:
            [self.jl_p, self.jl_s, self.jl_i] = np.zeros((3, self.veclength))            
    
    def helper_dxdy(self, x, y):
        delta = np.diff(y) / np.diff(x)
        return np.append(delta,delta[-1]) 
                   
    def vg(self, n, wl):
            return self.c / (n - wl * self.helper_dxdy(wl, n))
    
     
    def gen_jl(self, y):
        """ Following Eqn (8) in Seres & Hebling, "Nonstationary theory of 
            synchronously pumped femtosecond optical parametric oscillators", 
            JOSA B Vol 17 No 5, 2000. A call to this function updates the 
            :math: `\chi_3` mixing terms used for four-wave mixing.
            
            Parameters
            ----------
            y : array-like, shape is 3 * NPTS
                Concatenated pump, signal, and idler fields
            
            """
        n_p  = self.n_p
        n_s  = self.n_s
        n_i  = self.n_i
        
        vg_p = self.vg(n_p, self.pump.w_hz)
        vg_s = self.vg(n_s, self.sgnl.w_hz)
        vg_i = self.vg(n_i, self.idlr.w_hz)
        
        gamma_p = constants.epsilon_0 * 0.5 * n_p**2 * vg_p 
        gamma_s = constants.epsilon_0 * 0.5 * n_s**2 * vg_s
        gamma_i = constants.epsilon_0 * 0.5 * n_i**2 * vg_i
        
        jl = np.zeros((3, self.veclength), dtype = 'complex128')
        gamma = [gamma_p, gamma_s, gamma_i]
        waist = [self.pump_beam.waist, self.sgnl_beam.waist, self.idlr_beam.waist]
        
        i = 0               
        for vec1 in [self.Ap, self.As, self.Ai]:
            for vec2 in [self.Ap, self.As, self.Ai]:
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
        
    
    def poling(self, x):
        """ Helper function to get sign of :math: `d_\textrm{eff}` at position
            :math: `x` in the crystal. Uses self.crystal's pp function. 
            
            For APPLN this is somewhat complicated. The input position x could
            be many periods away from the previous value, and in either
            direction. One solution would be carefully stepping back and forth,
            but this needs to be perfect to prevent numerical errors. 
            
            Instead, precompute the domain boundaries and use a big comparison
            to check the poling(z)
            
            
            Returns
            -------
            x : int
                Sign (+1 or -1) of :math: `d_\textrm{eff}`.
            """
        if ((self.domain_lb < x) * (x < self.domain_ub)).any():
            return -1
        else:
            return 1

    def precompute_poling(self):
        z_current = 0
        domain_lb = []
        domain_ub = []
        while z_current < self.crystal.length_mks:
            domain_lb.append(z_current+self.crystal.pp(z_current) * 0.5)
            domain_ub.append(z_current+self.crystal.pp(z_current) * 1.0)
            z_current += self.crystal.pp(z_current)
            if self.crystal.pp(z_current) <= 1e-6:
                print("Error: poling period too small")
        self.domain_lb = np.array(domain_lb)
        self.domain_ub = np.array(domain_ub)
        plt.plot(self.domain_lb)
        plt.plot(self.domain_ub)
        
    
    
    def Ap(self, y):
        return y[0                  : self.veclength]
    def As(self, y):
        return y[self.veclength     : 2 * self.veclength]
    def Ai(self, y):
        return y[2 * self.veclength : 3 * self.veclength]
        
    # Integrand:
    # State is defined by:
    # 1.) fields in crystal
    # 2.) values of k
    # 3.) electric field->intensity conversion (~ area)

    # Output is vector of estimate for d/dz field
        
    def deriv(self, z, y, dydx):
        assert not np.isnan(z)
        if self.crystal.mode == 'PP':          
            deff = self.poling(z) * self.crystal.deff   
        elif self.crystal.mode == 'BPM' or self.crystal.mode == 'simple':
            deff = self.crystal.deff
        else:
            raise exceptions.AttributeError('Crystal type not known; deff not set.')
        # After epic translation of Dopri853 from Numerical Recipes' C++ to
        # native Python/NumPy, we can use complex numbers throughout:
        t = z / float(self.approx_pulse_speed)
        self.phi_p[:] = np.exp(1j * ((self.k_p + self.k_p_0) * z - t * self.pump.W_mks))
        self.phi_s[:] = np.exp(1j * ((self.k_s + self.k_s_0) * z - t * self.sgnl.W_mks))
        self.phi_i[:] = np.exp(1j * ((self.k_i + self.k_i_0) * z - t * self.idlr.W_mks))

                                        
        z_to_focus = z - self.crystal.length_mks/2.0
        if self._calc_gouy:
            self.phi_p *= self.pump_beam.calculate_gouy_phase(z_to_focus, self.n_p)
            self.phi_s *= self.sgnl_beam.calculate_gouy_phase(z_to_focus, self.n_s)
            self.phi_i *= self.idlr_beam.calculate_gouy_phase(z_to_focus, self.n_i)
        
        if not self.disable_SPM:
            self.gen_jl(y)

    
        if self._wg_mode == False:
            waist_p = self.pump_beam.calculate_waist(z_to_focus, n_s = self.n_p)
            waist_s = self.sgnl_beam.calculate_waist(z_to_focus, n_s = self.n_s)
            waist_i = self.idlr_beam.calculate_waist(z_to_focus, n_s = self.n_i)
            R_p = self.pump_beam.calculate_R(z_to_focus,  n_s = self.n_p)
            R_s = self.sgnl_beam.calculate_R(z_to_focus,  n_s = self.n_s)
            R_i = self.idlr_beam.calculate_R(z_to_focus,  n_s = self.n_i)

        # Geometric scaling factors (Not used)
        # P_to_a is the conversion between average power and "effective intensity"
        # The smallest area is used, as this is the part of the field which is 
        # interacting. THe larger fields must be scaled with a mode-match integral
        # THe mode-match integral-based scale factor for Gaussian beams is  
        #             4 * w1**2 w2**2
        #             ---------------
        #           (w1**2 + w2 **2)**2
        # This is the power coupling, so multiply larger 2 fields by sqrt(MMI)
        #
        # Attempting to limit interaction via mode-match integrals appears to
        # (1) not conserve energy and (2) INCREASES the interaction strength.
        # I'm not totally sure why, but it seems like the best course of action
        # is to match confocal parameters (which is done in __init__, above).
        # Overlap integrals are left intact, in case we want to plot them.

        if self._wg_mode == False:
            if (np.mean(waist_p) <= np.mean(waist_s)) and (np.mean(waist_p) <= np.mean(waist_i)):            
                self.pump_P_to_a = self.pump_beam.rtP_to_a_2(self.pump,self.crystal,z_to_focus)
                self.sgnl_P_to_a = self.sgnl_beam.rtP_to_a_2(self.sgnl, self.crystal, None, waist_p)
                self.idlr_P_to_a = self.idlr_beam.rtP_to_a_2(self.idlr, self.crystal, None, waist_p)
                self.overlap_pump = 1.0
                self.overlap_sgnl = np.sqrt(self.sgnl_beam.calc_overlap_integral(z_to_focus, self.sgnl, self.pump, self.pump_beam, self.crystal))
                self.overlap_idlr = np.sqrt(self.idlr_beam.calc_overlap_integral(z_to_focus, self.idlr, self.pump, self.pump_beam, self.crystal))
            elif np.mean(waist_s) <= np.mean(waist_i):         
                self.sgnl_P_to_a = self.sgnl_beam.rtP_to_a_2(self.sgnl,self.crystal,z_to_focus)
                self.pump_P_to_a = self.pump_beam.rtP_to_a_2(self.pump, self.crystal, None, waist_s)
                self.idlr_P_to_a = self.idlr_beam.rtP_to_a_2(self.idlr, self.crystal, None, waist_s)
                self.overlap_pump = np.sqrt(self.pump_beam.calc_overlap_integral(z_to_focus, self.pump, self.sgnl, self.sgnl_beam, self.crystal))
                self.overlap_sgnl = 1.0
                self.overlap_idlr = np.sqrt(self.idlr_beam.calc_overlap_integral(z_to_focus, self.idlr, self.sgnl, self.sgnl_beam, self.crystal))
            else:              
                self.idlr_P_to_a = self.idlr_beam.rtP_to_a_2(self.idlr,self.crystal,  None, waist_i)
                self.sgnl_P_to_a = self.sgnl_beam.rtP_to_a_2(self.sgnl, self.crystal, None, waist_i)
                self.pump_P_to_a = self.pump_beam.rtP_to_a_2(self.pump, self.crystal, None, waist_i)
                self.overlap_pump = np.sqrt(self.pump_beam.calc_overlap_integral(z_to_focus, self.pump, self.idlr, self.idlr_beam, self.crystal))
                self.overlap_sgnl = np.sqrt(self.sgnl_beam.calc_overlap_integral(z_to_focus, self.sgnl, self.idlr, self.idlr_beam, self.crystal))
                self.overlap_idlr = 1.0
    
            if self._plot_beam_overlaps and abs(z-self.last_calc_z) > self.crystal.length_mks*0.001:    
                plt.subplot(131)
                plt.plot(z*1e3, np.mean(self.overlap_pump), '.b')
                plt.plot(z*1e3, np.mean(self.overlap_sgnl), '.k')
                plt.plot(z*1e3, np.mean(self.overlap_idlr), '.r')
                plt.subplot(132)
                plt.plot(z*1e3, np.mean(waist_p)*1e6, '.b')
                plt.plot(z*1e3, np.mean(waist_s)*1e6, '.k')
                plt.plot(z*1e3, np.mean(waist_i)*1e6, '.r')
                plt.subplot(133)
                plt.plot(z*1e3, np.mean(R_p), '.b')
                plt.plot(z*1e3, np.mean(R_s), '.k')
                plt.plot(z*1e3, np.mean(R_i), '.r')
                self.last_calc_z = z
        else:
            # Life is simple in waveguide mode (for large V number WG)
            self.pump_P_to_a = self.pump_beam.rtP_to_a(self.n_p)
            self.sgnl_P_to_a = self.sgnl_beam.rtP_to_a(self.n_s)
            self.idlr_P_to_a = self.idlr_beam.rtP_to_a(self.n_i)
            
        self.AsAi[:] =  np.power(self.phi_p, -1.0)*\
            self.fftobject.conv(self.sgnl_P_to_a * self.As(y) * self.phi_s,
                                self.idlr_P_to_a * self.Ai(y) * self.phi_i)
            
        self.ApAs[:] =  np.power(self.phi_i, -1.0)*\
            self.fftobject.corr(self.pump_P_to_a * self.Ap(y) * self.phi_p,
                                self.sgnl_P_to_a * self.As(y) * self.phi_s)
            
        self.ApAi[:] =  np.power(self.phi_s, -1.0)*\
            self.fftobject.corr(self.pump_P_to_a * self.Ap(y) * self.phi_p, 
                                self.idlr_P_to_a * self.Ai(y) * self.phi_i)


        L = self.veclength        
    
        
        # np.sqrt(2 / (c * eps * pi * waist**2)) converts to electric field        
        #
        # From the Seres & Hebling paper,
        # das/dz + i k as = F(ap, ai)
        # The change of variables is as = As exp[-ikz], so that
        #
        # das/dz = dAs/dz exp[-ikz] - ik As exp[ikz]
        # das/dz + ik as = ( dAs/dz exp[-ikz] - ik As exp[-ikz] ) + i k As exp[-ikz]
        #                = dAs/dz exp[-ikz]
        # The integration is done in the As variables, to remove the fast k
        # dependent term. The procedure is:
        #   1) Calculate F(ai(Ai), ap(Ap))
        #   2) Multiply by exp[+ikz]
        
        # If the chi-3 terms are included:
        if not self.disable_SPM:
            logging.warn('Warning: this code not updated with correct field-are scaling. Fix it if you use it!')
            jpap = self.phi_p**-1 * self.fftobject.conv(self.jl_p, self.Ap(y) * self.phi_p) * \
                   np.sqrt(2. / (constants.speed_of_light * constants.epsilon_0 * np.pi * waist_p**2))
            jsas = self.phi_s**-1 * self.fftobject.conv(self.jl_s, self.As(y) * self.phi_s) * \
                   np.sqrt(2. / (constants.speed_of_light* constants.epsilon_0 * np.pi * waist_s**2))
            jiai = self.phi_i**-1 * self.fftobject.conv(self.jl_i, self.Ai(y) * self.phi_i) * \
                   np.sqrt(2. / (constants.speed_of_light* constants.epsilon_0 * np.pi * waist_i**2))      
                   
            dydx[0  :L  ] = 1j * 2 * self.AsAi * self.pump.W_mks * deff / (constants.speed_of_light* self.n_p) / \
                     self.pump_P_to_a -1j * self.pump.w_hz * self.crystal.n2 / (2.*np.pi*self.c) * jpap
            dydx[L  :2*L] = 1j * 2 * self.ApAi * self.sgnl.W_mks * deff / (constants.speed_of_light* self.n_s) / \
                    self.sgnl_P_to_a  -1j * self.sgnl.w_hz * self.crystal.n2 / (2.*np.pi*self.c) * jsas
            dydx[2*L:3*L] = 1j * 2 * self.ApAs * self.idlr.W_mks * deff / (constants.speed_of_light* self.n_i) / \
                    self.idlr_P_to_a  -1j * self.idler.w_hz * self.crystal.n2 / (2.*np.pi*self.c) * jiai
        else:
            # Only chi-2:
            # pump
            dydx[0  :L  ] = 1j * 2 * self.AsAi * self.pump.W_mks * deff / (constants.speed_of_light * self.n_p) / \
                    (self.pump_P_to_a) 
            # signal
            dydx[L  :2*L] = 1j * 2 * self.ApAi * self.sgnl.W_mks * deff / (constants.speed_of_light * self.n_s) / \
                    (self.sgnl_P_to_a) 
            # idler
            dydx[2*L:3*L] = 1j * 2 * self.ApAs * self.idlr.W_mks * deff / (constants.speed_of_light * self.n_i) / \
                    (self.idlr_P_to_a) 
    def process_stepper_output(self, solver_out):
        """ Post-process output of ODE solver.
        

        The saved data from an ODE solved are the pump, signal, and idler in
        the dispersionless reference frame. To see the pulses "as they really
        are", this dispersion must be added back in.
        
        Parameters
        ----------
        solver_out
            Output class instance from ODESolve
        Returns
        ---------
        dfg_results
            Instance of dfg_results_interface class
        """
        npoints = self.veclength
        
        pump_out = solver_out.ysave[0:solver_out.count, 0        :   npoints]
        sgnl_out = solver_out.ysave[0:solver_out.count, npoints  : 2*npoints]
        idlr_out = solver_out.ysave[0:solver_out.count, 2*npoints: 3*npoints]
        zs       = solver_out.xsave[0:solver_out.count]
        print 'Pulse velocity is ~ '+str(self.approx_pulse_speed*1e-12)+'mm/fs'
        print('ks: '+str(self.k_p_0)+' '+str(self.k_s_0)+' '+str(self.k_i_0))
        
        pump_pulse_speed = constants.speed_of_light / self.n_p[self._pump_center_idx]
                                  
        for i in xrange(solver_out.count):
            z = zs[i]
            print z
            t =  z / pump_pulse_speed

            phi_p = np.exp(1j * ((self.k_p + self.k_p_0) * z - t * self.pump.W_mks) )
            phi_s = np.exp(1j * ((self.k_s + self.k_s_0) * z - t * self.sgnl.W_mks))
            phi_i = np.exp(1j * ((self.k_i + self.k_i_0) * z - t * self.idlr.W_mks))


            pump_out[i, :] *= phi_p
            sgnl_out[i, :] *= phi_s
            idlr_out[i, :] *= phi_i

        interface = dfg_results_interface(self, pump_out, sgnl_out, idlr_out, zs)
        return interface
        
    def format_overlap_plots(self):
        plt.subplot(131)
        plt.ylabel('Overlap with smallest beam')
        plt.xlabel('Crystal length (mm)')
        plt.subplot(132)
        plt.ylabel('Beam waist (um)')
        plt.xlabel('Crystal length (mm)')
        plt.subplot(133)
        plt.ylabel('Beam curvature (m)')
        plt.xlabel('Crystal length (mm)')            

class dfg_results_interface:
    """
        Interface to output of DFG solver. This class provides a clean way
        of working with the DFG field using the Pulse class. 
        
        Notes
        -----
        After initialization, calling::
            
                get_{pump,sgnl,idlr}(n)
        
        will set the dfg results class' "pulse" instance to the appropriate
        field and return it.
        
        Example
        -------
        To plot the 10th saved signal field, you would call::
                
                p = dfg_results_interface.get_sgnl(10-1)
                plt.plot(p.T_ps, abs(p.AT)**2 )
        
        To get the actual position (z [meters]) that this corresponds to,
        call::
                
                z = dfg_results_interface.get_z(10-1)
        
"""
    n_saves = 0
    pump_field = []
    sgnl_field = []
    idlr_field = []
    
    def __init__(self, integrand_instance, pump, sgnl, idlr, z):        
        self.pulse = integrand_instance.pump.create_cloned_pulse()
        
        self.pump_wl = integrand_instance.pump.center_wavelength_nm
        self.sgnl_wl = integrand_instance.sgnl.center_wavelength_nm
        self.idlr_wl = integrand_instance.idlr.center_wavelength_nm
        
        self.pump_field = pump[:]
        self.sgnl_field = sgnl[:]
        self.idlr_field = idlr[:]

        self.pump_max_field = np.max(abs(pump))
        self.sgnl_max_field = np.max(abs(sgnl))
        self.idlr_max_field = np.max(abs(idlr))
        

        self.pump_max_temporal = np.max(abs(np.fft.fft(pump)))
        self.sgnl_max_temporal = np.max(abs(np.fft.fft(sgnl)))
        self.idlr_max_temporal = np.max(abs(np.fft.fft(idlr)))
        
        self.zs         = z[:]
        self.n_saves = len(z)
        print('wls: '+str(self.pump_wl)+' '+str(self.sgnl_wl)+' '+str(self.idlr_wl))
        
    def get_z(self, n):
        return self.zs[n]
        
    def get_pump(self, n):
        self.pulse.set_AW(self.pump_field[n])
        self.pulse.set_center_wavelength_nm(self.pump_wl)
        return self.pulse
                
    def get_sgnl(self, n):
        self.pulse.set_AW(self.sgnl_field[n])
        self.pulse.set_center_wavelength_nm(self.sgnl_wl)
        return self.pulse
        
    def get_idlr(self, n):
        self.pulse.set_AW(self.idlr_field[n])
        self.pulse.set_center_wavelength_nm(self.idlr_wl)
        return self.pulse
        
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
