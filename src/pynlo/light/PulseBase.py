# -*- coding: utf-8 -*-
#Created on Thu Jun 04 13:48:11 2015
#This file is part of pyNLO.
#
#    pyNLO is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    pyNLO is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with pyNLO.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from scipy.interpolate import interp1d
from scipy import constants, signal 
from pynlo.util import FFT_t, IFFT_t
import exceptions
import warnings

class Pulse:
    """Class which carried all information about the light field. This class 
       is a base upon which various cases are built (eg analytic pulses,
       CW fields, or pulses generated from experimental data.) """

    def __init__(self, frep_MHz = None, n = None):
        if frep_MHz is not None:
            self._frep_MHz = frep_MHz
            if frep_MHz > 1.0e6:
                warnings.warn("frep should be specified in MHz; large value given.")
        if n is not None:
            self.set_NPTS(n)            
        # Constants, moved here so that module runs through Sphynx autodoc when
        # scipy is Mocked out.
        self._c_nmps = constants.value('speed of light in vacuum')*1e9/1e12 # c in nm/ps
        self._c_mks  = constants.value('speed of light in vacuum') # m/s        
    # Private variables:
    # This set is the minimum number required to completely specify the light
    # field. All other representations are derived from them.
    _n                  = 0       # Number of points on grid
    _centerfrequency    = 1.0     # Center frequency (THz)
    _time_window        = 1.0     # Time window (ps)
    _V                  = None    # Relative angular frequency grid (2 pi THz)
    _AW                 = None    # Frequency-domain pulse amplitude
    _frep_MHz           = 100.0   # Pulse frequency (MHz); used for converting
                                  # pulse energy < - > average power
    _ready              = False   # All fields are initialized (this allows for
                                  # incomplete Pulse objects to be created and
                                  # and filled in later)         
    _external_units     = None
    # Constants
    _c_nmps = None
    _c_mks  = None
    # Cached values for expensive functions that I have identified as widely-used
    # in a profiler. Note that this is a sparse list...
    # Wavelength
    _cache_wl_nm_hash               = None
    _cache_wl_nm                    = None
    # Frequency in THz
    _cache_W_Hz_hash                = None
    _cache_W_Hz                     = None
    
    _not_ready_msg = 'Pulse class is not yet ready -- set center wavelength, time window, and npts.'    
    ####### Private properties    #############################################
    def __get_w0(self):
        r""" Return center angular frequency (THz) """
        if self._centerfrequency is None:
            raise exceptions.ValueError('Center frequency is not set.')
        return 2.0 * np.pi * self._centerfrequency    
    def __get_W(self):
        r""" Return angular frequency grid (THz) """
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        else:
            return self._V + self._w0
    def __get_T(self):
        r""" Return temporal grid (ps) """
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        else:
            TGRID =  np.linspace(-self._time_window / 2.0,
                                  self._time_window / 2.0,
                                  self._n, endpoint = False) # time grid
            return TGRID
    def __get_dT(self):
        r""" Return time grid spacing (ps) """
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        else:
            return self._time_window / np.double(self._n)

    def __get_V(self):
        r""" Return relative angular frequency grid (THz) """
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        else:
            VGRID = 2.0*np.pi*np.transpose(np.arange(-self._n/2,
                                                      self._n/2))/(self._n*self._dT) # Frequency grid (angular THz)        
            return VGRID

            
    _w0     = property(__get_w0)        
    _W      = property(__get_W)
    _dT     = property(__get_dT)                          
    _T      = property(__get_T)
    _dT     = property(__get_dT)
    _V      = property(__get_V)

    ####### Public  properties    #############################################    
    # The following are properties. They should be used by other codes to
    # provide some isolation from the Pulse class's internal mechanics.
    
    # Wavelength is dynamically derived from frequency grid
    def _get_wavelength_nm(self):
        if (self.cache_hash == self._cache_wl_nm_hash):
           return self._cache_wl_nm
        else:
           self._cache_wl_nm_hash = self.cache_hash
           self._cache_wl_nm = 2*np.pi*self._c_nmps / self.W_THz
           return self._cache_wl_nm
    # Wavelength is dynamically derived from frequency grid
    def _get_wavelength_m(self):    
        return 2*np.pi*self._c_mks / self.W_mks
    def _get_center_wavelength_nm(self):
        return self._c_nmps / self._centerfrequency
    def _get_center_wavelength_mks(self):
        return (self._c_nmps / self._centerfrequency )*1.0e9
    def _get_center_frequency_THz(self):
        return self._centerfrequency        
    def _get_center_frequency_mks(self):
        return self._centerfrequency * 1.0e12
        
    def _get_NPTS(self):
        return self._n
    def _get_hash(self):
        return str(self._centerfrequency)+str(self.NPTS)    
    def _get_W_Hz(self):
        if (self._cache_W_Hz_hash == self.cache_hash):
            return self._cache_W_Hz
        else:
            self._cache_W_Hz_hash = self.cache_hash
            self._cache_W_Hz                 =  self._W * 1e12
            return self._cache_W_Hz
    def _get_F_Hz(self):
        return self._get_W_Hz() / (2.0*np.pi)            
    def _get_W_THz(self):
        return self._W
    def _get_F_THz(self):
        return self._W / (2.0*np.pi)
    def _get_dT_seconds(self):
        return self._dT * 1e-12
    def _get_dT_picoseconds(self):
        return self._dT
        
    def _get_T_seconds(self):
        return self._T* 1e-12
    def _get_T_picoseconds(self):
        return self._T

    def _get_time_window_seconds(self):
        return self._time_window* 1e-12
    def _get_time_window_picoseconds(self):
        return self._time_window

    def _get_V_Hz(self):
        return self._V* 1e12
    def _get_V_THz(self):
        return self._V
    def _get_dF_THz(self):
        return abs(self.W_THz[1]-self.W_THz[0])/(2.0*np.pi)
    def _get_dF_Hz(self):
        return abs(self.W_mks[1]-self.W_mks[0])/(2.0*np.pi)
    def _get_frep_MHz(self):
        return self._frep_MHz
    def _get_frep_Hz(self):
        if self._frep_MHz is None:
            return None
        else:
            return self._frep_MHz * 1.0e6
    def _get_AW(self):
        if self._AW is not None:
            return self._AW.copy()
        else:
            raise exceptions.RuntimeError('Grids not yet set up.')
    def _get_AT(self):        
        return IFFT_t( self._AW.copy() )
    
    def set_AW(self, AW_new):
        r""" Set the value of the frequency-domain electric field.
        
        Parameters
        ----------
        AW_new : array_like
            New electric field values. 
        
        """
        if not self._ready:
            raise exceptions.RuntimeError(self._not_ready_msg)
        if self._AW is None:
            self._AW = np.zeros((self._n,), dtype = np.complex128)
        self._AW[:] = AW_new
        
    def set_AT(self, AT_new):
        r""" Set the value of the time-domain electric field.
        
        Parameters
        ----------
        AW_new : array_like
            New electric field values.
            
        """
        self.set_AW( FFT_t(AT_new ))

    # To keep this class' working isolated from accessors, all data reading and
    # writing is done via methods. These are:

    wl_nm           = property(_get_wavelength_nm)
    """ Property: Wavelength grid
        
    Returns
    -------
    wl_nm : ndarray, shape NPTS
        Wavelength grid corresponding to AW [nm]
    """


    W_THz           = property(_get_W_THz)
    """ Property: angular frequency grid 
        
    Returns
    -------
    W_THz : ndarray, shape NPTS
        Angular frequency grid corresponding to AW [THz]
    """

    F_THz           = property(_get_F_THz)
    """ Property: frequency grid 
        
    Returns
    -------
    F_THz : ndarray, shape NPTS
        Frequency grid corresponding to AW [THz]
    """

    dT_ps           = property(_get_dT_picoseconds)
    """    
    Property: time grid spacing
    
    Returns
    -------
    dT_ps : float
        Time grid spacing [ps]
    """    
    
    T_ps            = property(_get_T_picoseconds)
    """    
    Property: time grid
    
    Returns
    -------
    T_ps : ndarray, shape NPTS
        Time grid corresponding to AT [ps]
    """
    
    V_THz           = property(_get_V_THz)
    """    
    Property: relative angular frequency grid
    
    Returns
    -------
    V_THz : ndarray, shape NPTS
        Relative angular frequency grid corresponding to AW [THz]
    """
    
    time_window_ps  = property(_get_time_window_picoseconds)
    """    
    Property: time grid span
    
    Returns
    -------
    time_window_ps : float
        Time grid span [ps]
    """    
    
    center_wavelength_nm    = property(_get_center_wavelength_nm)
    """    
    Property: center wavelength
    
    Returns
    -------
    center_wavelength_nm : float
        Wavelength of center point in AW grid [nm]
    """        
    center_frequency_THz = property(_get_center_frequency_THz)
    """    
    Property: center frequency
    
    Returns
    -------
    center_frequency_THz : float
        Frequency of center point in AW grid [THz]
    """        
    
    wl_mks          = property(_get_wavelength_m)
    """ Property: Wavelength grid
        
    Returns
    -------
    wl_mks : ndarray, shape NPTS
        Wavelength grid corresponding to AW [m]
    """    
    W_mks           = property(_get_W_Hz)
    """ Property: angular frequency grid 
        
    Returns
    -------
    W_mks : ndarray, shape NPTS
        Angular frequency grid corresponding to AW [Hz]
    """    
    F_mks           = property(_get_F_Hz)
    """ Property: frequency grid 
        
    Returns
    -------
    F_mks : ndarray, shape NPTS
        Frequency grid corresponding to AW [Hz]
    """    

    dT_mks          = property(_get_dT_seconds)
    """    
    Property: time grid spacing
    
    Returns
    -------
    dT_mks : float
        Time grid spacing [s]
    """    
    
    T_mks           = property(_get_T_seconds)
    """    
    Property: time grid
    
    Returns
    -------
    T_mks : ndarray, shape NPTS
        Time grid corresponding to AT [s]
    """
    
    V_mks           = property(_get_V_Hz)
    """    
    Property: relative angular frequency grid
    
    Returns
    -------
    V_mks : ndarray, shape NPTS
        Relative angular frequency grid corresponding to AW [Hz]
    """    
    dF_mks           = property(_get_dF_Hz) 
    """    
    Property: frequency grid spacing
    
    Returns
    -------
    dF_mks : float
        Frequency grid spacing [s]
    """     
    dF_THz           = property(_get_dF_THz)
    """    
    Property: frequency grid spacing
    
    Returns
    -------
    dF_ps : float
        Frequency grid spacing [ps]
    """     
    
    time_window_mks = property(_get_time_window_seconds)
    """    
    Property: time grid span
    
    Returns
    -------
    time_window_mks : float
        Time grid span [ps]
    """    
    
    center_wavelength_mks   = property(_get_center_wavelength_mks)
    """    
    Property: center wavelength
    
    Returns
    -------
    center_wavelength_mks : float
        Wavelength of center point in AW grid [m]
    """            
    
    center_frequency_mks = property(_get_center_frequency_mks)
    """    
    Property: center frequency
    
    Returns
    -------
    center_frequency_mks : float
        Frequency of center point in AW grid [Hz]
    """            
    
    AW              = property(_get_AW)
    """    
    Property: frequency-domain electric field grid
    
    Returns
    -------
    AW : ndarray, shape NPTS
        Complex electric field in frequency domain.
    """
    
    AT              = property(_get_AT)
    """    
    Property: time-domain electric field grid
    
    Returns
    -------
    AT : ndarray, shape NPTS
        Complex electric field in time domain.
    """
    NPTS            = property(_get_NPTS)  
    frep_MHz        = property(_get_frep_MHz)
    """    
    Property: Repetition rate. Used for calculating average beam power.
    
    Returns
    -------
    frep_MHz : float
        Pulse repetition frequency [MHz]
    """                
    frep_mks         = property(_get_frep_Hz)
    """    
    Property: Repetition rate. Used for calculating average beam power.
    
    Returns
    -------
    frep_mks : float
        Pulse repetition frequency [Hz]
    """                
    cache_hash      = property(_get_hash)
    
    def _set_centerfrequency(self, f_THz):
        self._centerfrequency = f_THz
        self._check_ready()
    def _set_time_window(self, T_ps):
        self._time_window = T_ps
        self._check_ready()

    def _check_ready(self):
        self._ready =  (self._centerfrequency is not None) and\
                       (self._n is not None) and\
                       (self._time_window is not None)
               

    def _ext_units_nmps(self):
        if self._external_units is None:
            exceptions.RuntimeError('Unit type has not been set.')
        return self._external_units == 'nmps'
    def _ext_units_mks(self):
        if self._external_units is None:
            exceptions.RuntimeError('Unit type has not been set.')
        return self._external_units == 'mks'

    ####### Core public  functions     ########################################        
    def set_center_wavelength_nm(self, wl):
        r""" Set the center wavelength of the grid in units of nanometers.
        
        Parameters
        ----------
        wl : float
             New center wavelength [nm]
        
        """
        self._set_centerfrequency(self._c_nmps / wl)
    def set_center_wavelength_m(self, wl):
        r""" Set the center wavelength of the grid in units of meters.
        
        Parameters
        ----------
        wl : float
             New center wavelength [m]
        
        """
        self._set_centerfrequency(self._c_nmps /  (wl * 1.0e9) )
        
    def set_NPTS(self, NPTS):
        r""" Set the grid size. 
        
        The actual grid arrays are *not* altered
        automatically to reflect a change.
        
        Parameters
        ----------
        NPTS : int
             Number of points in grid
        
        """        
        self._n = int(NPTS)
        self._check_ready() 
    def set_frep_MHz(self, fr_MHz):
        r""" Set the pulse repetition frequency. 
        
        This parameter used internally to convert between pulse energy and 
        average power.
        
        Parameters
        ----------
        fr_MHz : float
             New repetition frequency [MHz]
        
        """        
        self._frep_MHz = fr_MHz
    def set_time_window_ps(self, T):
        r""" Set the total time window of the grid. 
        
        This sets the grid dT, and
        implicitly changes the frequency span (~1/dT).
        
        Parameters
        ----------
        T : float
             New grid time span [ps]
        
        """                
        if self._n is None:
            raise exceptions.RuntimeError('Set number of points before setting time window.')
        # frequency grid is 2 pi/ dT * [-1/2, 1/2]
        # dT is simply time_window / NPTS
        self._set_time_window(T)
    def set_time_window_s(self, T):
        r""" Set the total time window of the grid. 
        
        This sets the grid dT, and
        implicitly changes the frequency span (~1/dT).
        
        Parameters
        ----------
        T : float
             New grid time span [s]        
        """                
        if self._n is None:
            raise exceptions.RuntimeError('Set number of points before setting time window.')        
        self._set_time_window(T * 1e12)
        
    def set_frequency_window_THz(self, DF):
        r""" Set the total frequency window of the grid. 
        
        This sets the grid dF, and
        implicitly changes the temporal span (~1/dF).
        
        Parameters
        ----------
        DF : float
             New grid time span [THz]
        
        """                
        if self._n is None:
            raise exceptions.RuntimeError('Set number of points before setting frequency window.')
        # Internally, the time window is used to determine the grids. Calculate
        # the time window size as  1 / dF = 1 / (DF / N)
        T = self._n / float(DF)
        self._set_time_window(T)
    def set_frequency_window_mks(self, DF):
        r""" Set the total frequency window of the grid. 
        
        This sets the grid dF, and
        implicitly changes the temporal span (~1/dF).
        
        Parameters
        ----------
        DF : float
             New grid time span [Hz]
        
        """                
        if self._n is None:
            raise exceptions.RuntimeError('Set number of points before setting frequency window.')
        # Internally, the time window is used to determine the grids. Calculate
        # the time window size as  1 / dF = 1 / (DF / N)
        T = self._n / float(DF)
        self._set_time_window(T * 1e12)                            

    ####### Auxiliary public  functions     ###################################
    def calc_epp(self):
        r""" Calculate and return energy per pulse via numerical integration
            of :math:`A^2 dt`
            
            Returns
            -------
            x : float
                Pulse energy [J]
            """
        return self.dT_mks * np.trapz(abs(self.AT)**2)
    
    def set_epp(self, desired_epp_J):
        r""" Set the energy per pulse (in Joules)
            
            Parameters
            ----------
            desired_epp_J : float
                 the value to set the pulse energy [J]
                 
            Returns
            -------
            nothing
            """
        self.set_AT(self.AT * np.sqrt( desired_epp_J / self.calc_epp() ) )
        
    
    def add_noise(self, noise_type='sqrt_N_freq'):
        r""" 
         Adds random intensity and phase noise to a pulse. 
        
        Parameters
        ----------
        noise_type : string
             The method used to add noise. The options are: 
    
             ``sqrt_N_freq`` : which adds noise to each bin in the frequency domain, 
             where the sigma is proportional to sqrt(N), and where N
             is the number of photons in each frequency bin. 
    
             ``one_photon_freq``` : which adds one photon of noise to each frequency bin, regardless of
             the previous value of the electric field in that bin. 
             
        Returns
        -------
        nothing
        """
        
        # This is all to get the number of photons/second in each frequency bin:
        size_of_bins = self.dF_mks                          # Bin width in [Hz]
        power_per_bin = np.abs(self.AW)**2 * size_of_bins  # [W/Hz]  * [Hz]
            
        h = constants.Planck # use scipy's constants package
        
        #photon_energy = h * self.W_THz/(2*np.pi) * 1e12
        photon_energy = h * self.F_mks # h nu
        photons_per_bin = power_per_bin/photon_energy # photons / second
        photons_per_bin[photons_per_bin<0] = 0 # must be positive.
        print np.sum(np.sqrt(photons_per_bin))
        print photons_per_bin.shape
        
        # now generate some random intensity and phase arrays:
        size = np.shape(self.AW)[0]
        random_intensity = np.random.normal(size=size)
        random_phase = np.random.uniform(size=size) * 2 * np.pi
        
        if noise_type == 'sqrt_N_freq': # this adds Gausian noise with a sigma=sqrt(photons_per_bin)
            noise = random_intensity * np.sqrt(photons_per_bin) * photon_energy * size_of_bins * 1e12 * np.exp(1j*random_phase)
        
        elif noise_type == 'one_photon_freq': # this one photon per bin in the frequecy domain
            noise = random_intensity * photon_energy * size_of_bins * 1e12 * np.exp(1j*random_phase)
        else:
            raise ValueError('noise_type not recognized. So far only sqrt_N_freq is supported')
        
        self.set_AW(self.AW + noise)
        
        
        
    def chirp_pulse_W(self, GDD, TOD=0, FOD = 0.0, w0_THz = None):
        r""" Alter the phase of the pulse 
        
        Apply the dispersion coefficients :math:`\beta_2, \beta_3, \beta_4`
        expanded around frequency :math:`\omega_0`.
        
        Parameters
        ----------
        GDD : float
             Group delay dispersion (:math:`\beta_2`) [ps^2]
        TOD : float, optional
             Group delay dispersion (:math:`\beta_3`) [ps^3], defaults to 0.
        FOD : float, optional
             Group delay dispersion (:math:`\beta_4`) [ps^4], defaults to 0.             
        w0_THz : float, optional
             Center frequency of dispersion expansion, defaults to grid center frequency.
        
        Notes
        -----
        The convention used for dispersion is
        
        .. math:: E_{new} (\omega) = \exp\left(i \left(
                                        \frac{1}{2} GDD\, \omega^2 +
                                        \frac{1}{6}\, TOD \omega^3 +
                                        \frac{1}{24} FOD\, \omega^4 \right)\right)
                                        E(\omega)
                                            
        """                

        if w0_THz is None:
            self.set_AW( np.exp(1j * (GDD / 2.0) * self.V_THz**2 + 
                                   1j * (TOD / 6.0) * self.V_THz**3+ 
                                   1j * (FOD / 24.0) * self.V_THz**4) * self.AW )
        else:
            V = self.W_THz - w0_THz
            self.set_AW( np.exp(1j * (GDD / 2.0) * V**2 + 
                                   1j * (TOD / 6.0) * V**3+ 
                                   1j * (FOD / 24.0) * V**4) * self.AW )
    def apply_phase_W(self, phase):
        self.set_AW(self.AW * np.exp(1j*phase))
    def chirp_pulse_T(self, chirp2, chirp3, T0):
        self.set_AT( self.AT * np.exp(-1j * (chirp2 / 2.0) * (self.T_ps/T0)**2 + 
                                 -1j * (chirp3 / 3.0) * (self.T_ps/T0)**3) )
                                 
    def dechirp_pulse(self, GDD_TOD_ratio = 0.0, intensity_threshold = 0.05):

        spect_w = self.AW
        phase   = np.unwrap(np.angle(spect_w))
        ampl    = np.abs(spect_w)
        mask = ampl**2 > intensity_threshold * np.max(ampl)**2
        gdd     = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 2))
        self.set_AW( ampl * np.exp(1j*(phase-gdd(self.W_THz))) )
    def remove_time_delay(self, intensity_threshold = 0.05):

        spect_w = self.AW
        phase   = np.unwrap(np.angle(spect_w))
        ampl    = np.abs(spect_w)
        mask = ampl**2 > (intensity_threshold * np.max(ampl)**2)        
        ld     = np.poly1d(np.polyfit(self.W_THz[mask], phase[mask], 1))
        self.set_AW( ampl * np.exp(1j*(phase-ld(self.W_THz))) )
    def add_time_offset(self, offset_ps):
        """Shift field in time domain by offset_ps picoseconds. A positive offset
           moves the pulse forward in time. """
        phase_ramp = np.exp(-1j*self.W_THz*offset_ps)
        self.set_AW(self.AW * phase_ramp)
    def expand_time_window(self, factor_log2, new_pts_loc = "before"):
        r""" Expand the time window by zero padding.
        Parameters
        ----------
        factor_log2 : integer
            Factor by which to expand the time window (1 = 2x, 2 = 4x, etc.)
        new_pts_loc : string
            Where to put the new points. Valid options are "before", "even", 
            "after
        """
        num_new_pts = self.NPTS*(2**factor_log2 - 1)
        AT_current = self.AT
        
        self.set_NPTS(self.NPTS * 2**factor_log2)        
        self.set_time_window_ps(self.time_window_ps * 2**factor_log2)
        self._AW = None # Force generation of new array
        if new_pts_loc == "before":
            self.set_AT(np.hstack( (np.zeros(num_new_pts,), AT_current) ))
        elif new_pts_loc == "after":
            self.set_AT(np.hstack( (AT_current, np.zeros(num_new_pts,)) ))            
        elif new_pts_loc == "even":
            pts_before = int(np.floor(num_new_pts * 0.5))
            pts_after  = num_new_pts - pts_before
            self.set_AT(np.hstack( (np.zeros(pts_before,), 
                                    AT_current, 
                                    np.zeros(pts_after,)) ))            
        else:
            raise ValueError("new_pts_loc must be one of 'before', 'after', 'even'")
    def rotate_spectrum_to_new_center_wl(self, new_center_wl_nm):
        """Change center wavelength of pulse by rotating the electric field in
            the frequency domain. Designed for creating multiple pulses with same
            gridding but of different colors. Rotations is by integer and to
            the closest omega."""
        new_center_THz = self._c_nmps/new_center_wl_nm
        rotation = (self.center_frequency_THz-new_center_THz)/self.dF_THz
        self.set_AW(np.roll(self.AW, -1*int(round(rotation))))
    def interpolate_to_new_center_wl(self, new_wavelength_nm):
        r""" Change grids by interpolating the electric field onto a new
        frequency grid, defined by the new center wavelength and the current
        pulse parameters. This is useful when grid overlaps must be avoided,
        for example in difference or sum frequency generation.
                
        Parameters
        ----------
        new_wavelength_nm : float
             New center wavelength [nm]
        Returns
        -------
        Pulse instance        
        """                
        working_pulse = self.create_cloned_pulse()
        working_pulse.set_center_wavelength_nm(new_wavelength_nm)
        interpolator = interp1d(self.W_mks, self. AW,
                                bounds_error = False,
                                fill_value = 0.0)
        working_pulse.set_AW(interpolator(working_pulse.W_mks))
        return working_pulse
        
    def filter_by_wavelength_nm(self, lower_wl_nm, upper_wl_nm):
        AW_new = self.AW
        AW_new[self.wl_nm < lower_wl_nm] = 0.0
        AW_new[self.wl_nm > upper_wl_nm] = 0.0
        self.set_AW(AW_new)        
    def clone_pulse(self, p):
        '''Copy all parameters of pulse_instance into this one'''
        self.set_NPTS(p.NPTS)
        self.set_time_window_ps(p.time_window_ps)
        self.set_center_wavelength_nm(p.center_wavelength_nm)
        self._frep_MHz = p.frep_MHz
        self.set_AT(p.AT)
    def create_cloned_pulse(self):
        '''Create and return new pulse instance identical to this instance.'''
        p = Pulse()
        p.clone_pulse(self)
        return p               
    def create_subset_pulse(self, center_wl_nm, NPTS):
        """ Create new pulse with smaller frequency span, centered at closest 
            grid point to center_wl_nm, with NPTS grid points and
            frequency-grid values from this pulse. """
            
        if NPTS >= self.NPTS:
            raise exceptions.ValueError("New pulse must have fewer points than existing one.")
        p = Pulse()
        center_idx = np.argmin(abs(self.wl_nm - center_wl_nm))

        # We want to reduce the frequency span, which means holding df fixed
        # while reducing NPTS. The time window is 1/df, so this means holding
        # the time window fixed as well.

        p._frep_MHz = self.frep_MHz
        p.set_center_wavelength_nm(self.wl_nm[center_idx] )
        p.set_time_window_ps(self.time_window_ps)
        p.set_NPTS(NPTS)
        idx1 = center_idx - (NPTS >> 1)
        idx2 = center_idx + (NPTS >> 1)
        p.set_AW(self.AW[idx1:idx2])
        return p
        
    def calculate_weighted_avg_frequency_mks(self):
        avg = np.sum(abs(self.AW)**2 * self.W_mks)
        weights = np.sum(abs(self.AW)**2)
        result = avg / (weights * 2.0 * np.pi)
        return result
    def calculate_weighted_avg_wavelength_nm(self):
        return 1.0e9 * self._c_mks / self.calculate_weighted_avg_frequency_mks()
    def calculate_intensity_autocorrelation(self):  
        r""" Calculates and returns the intensity autocorrelation,  
        :math:`\int P(t)P(t+\tau) dt` 
        
        Returns
        -------
        x : ndarray, shape N_pts
            Intensity autocorrelation. The grid is the same as the pulse class'
            time grid.
            
        """  
        return np.correlate(abs(self.AT)**2, abs(self.AT), mode='same')  
    def write_frog(self,
                 fileloc = 'broadened_er_pulse.dat', # default EDFA spectrum
                 flip_phase = True):
        """Save pulse in FROG data format. Grid is centered at wavelength
        center_wavelength (nm), but pulse properties are loaded from data
        file. If flip_phase is true, all phase is multiplied by -1 [useful
        for correcting direction of time ambiguity]. time_window (ps) sets 
        temporal grid size. 
        
        power sets the pulse energy:
        if power_is_epp is True  then the number is pulse energy [J] 
        if power_is_epp is False then the power is average power [W], and 
        is multiplied by frep to calculate pulse energy"""       
        
        self.fileloc   = fileloc             
        phase_data = np.unwrap(np.angle(self.get_AW()))
        inten_data = np.abs(self.get_AW())
        wavel_data = self.internal_wl_to_nm(self.wl_nm)
        
        # Write pulse data file
        np.savetxt(self.fileloc, np.vstack((wavel_data, inten_data, phase_data)).T) 
    
    def spectrogram(self, gate_function_width_ps=0.050, time_steps=500):
        """
        This calculates a spectrogram, essentially showing the spectrum
        as a funcition of time delay. See Dudley Fig. 10, on p1153 for a description
        of the spectrogram in the context of supercontinuum generaiton. 
        (http://dx.doi.org/10.1103/RevModPhys.78.1135)
        
        
        Parameters
        ----------
        
        gate_function_width : float
            the width of the gate function in seconds. Typically something like 
            0.050 ps (50 fs) is used
        
        time_steps : int
            the number of delay time steps to use. More steps makes a higher 
            resolution spectrogram, but takes longer to process and plots.
            ~500 seems about right.
        
        
        Returns
        -------
        DELAYS : 2D numpy meshgrid 
            the columns have increasing delay (in ps)
        FREQS : 2D numpy meshgrid
            the rows have increasing frequency (in THz)
        spectrogram : 2D numpy array
            Following the convention of Dudley, the frequency runs along the y-axis
            (axis 0) and the time runs alon the x-axis (axis 1)
        
        
        Example
        -------
        
        The spectrogram can be visualized using something like this: ::
        
            plt.figure()
            DELAYS, FREQS, extent, spectrogram = pulse.spectrogram()
            plt.imshow(spectrogram, aspect='auto', extent=extent)
            plt.xlabel('Time (ps)')
            plt.ylabel('Frequency (THz)')
            plt.tight_layout
    
            plt.show()
        
        output:
        
        .. image:: https://cloud.githubusercontent.com/assets/1107796/13677657/25075ea4-e6a8-11e5-98b4-7813fa9a6425.png
           :width: 500px
           :alt: example_result
            
    
        """

        def gauss(x, A=1, mu=0, sigma=1): # gaussian function
            return A*np.exp(-(x-mu)**2/(2.*sigma**2))
            
        t = self.T_ps # working in ps
        
        delay = np.linspace(np.min(t), np.max(t), time_steps)
        D, T = np.meshgrid(delay, t)
        D, AT = np.meshgrid(delay, self.AT)
        
        phase = np.unwrap(np.angle(AT))
        amp   = np.abs(AT)
        
        # make a 2D array of E(time, delay)
        E = amp * np.cos(2 * np.pi * T * self.center_frequency_THz + phase) * \
            gauss(T, mu=D, sigma=gate_function_width_ps) # gate function
        
        spectrogram = np.fft.fft(E, axis=0)
        freqs = np.fft.fftfreq(np.shape(E)[0], t[1]-t[0])
        
        DELAYS, FREQS = np.meshgrid(delay, freqs)
        
        # just take positive frequencies:
        h = np.shape(spectrogram)[0]
        spectrogram = spectrogram[:h/2]
        DELAYS      = DELAYS[:h/2]
        FREQS       = FREQS[:h/2]
                
        # calculate the extent to make it easy to plot:
        extent = (np.min(DELAYS), np.max(DELAYS), np.min(FREQS), np.max(FREQS))
        
        return DELAYS, FREQS, extent, np.abs(spectrogram)
