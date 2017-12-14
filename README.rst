pyNLO: Nonlinear optics modeling for Python
===========================================

This README is best viewed at http://pynlo.readthedocs.io/en/latest/readme_link.html

Complete documentation is available at http://pynlo.readthedocs.io/

.. image:: https://cloud.githubusercontent.com/assets/1107796/13850062/17f09ea8-ec1e-11e5-9311-b94df29c01cb.png
   :width: 330px
   :alt: PyNLO
   :align: right


Introduction
------------

PyNLO provides an easy-to-use, object-oriented set of tools for modeling the nonlinear interaction of light with materials. It provides many functionalities for representing pulses of light, beams of light, and nonlinear materials, such as crystals and fibers. Also, it features methods for simulating both three-wave-mixing processes (such as DFG), as well as four-wave-mixing processes such as supercontinuum generation. 

Features:
	- A solver for the propagation of light through a Chi-3 material, useful for simulation pulse compression and supercontinuum generation in an optical fiber. This solver is highly efficient, thanks to an adaptive-step-size implementation of the "Fourth-order Runge-Kutta in the Interaction Picture " (RK4IP) method of `Hult (2007) <https://www.osapublishing.org/jlt/abstract.cfm?uri=jlt-25-12-3770>`_.
	
	- A solver for simulating Chi-2 processes such as difference frequency generation.
	
	- A flexible object-oriented system for treating laser pulses, beams, fibers, and crystals.
	
	- ...and much more!


Installation
------------

PyNLO requires Python 2, and is tested on Python 2.7 (Python 3 compatibility is a work-in-progress). If you don't already have Python, we recommend an "all in one" Python package such as the `Anaconda Python Distribution <https://www.continuum.io/downloads>`_, which is available for free.

With pip
~~~~~~~~

The latest "official release" can be installed from PyPi with ::

    pip install pynlo
	
The up-to-the-minute latest version can be installed from GitHub with ::

    pip install git+https://github.com/pyNLO/PyNLO.git


With setuptools
~~~~~~~~~~~~~~~

Alternatively, you can download the latest version from the `PyNLO Github site <https://github.com/pyNLO/PyNLO>`_ (look for the "download zip" button), `cd` to the PyNLO directory, and use ::

    python setup.py install

Or, if you wish to edit the PyNLO source code without re-installing each time ::

    python setup.py develop


Documentation
-------------
The complete documentation for PyNLO is availabe at https://pynlo.readthedocs.org.


Example of use
--------------

The following example demonstrates how to use PyNLO to simulate the propagation of a 50 fs pulse through a nonlinear fiber using the split-step Fourier model (SSFM). Note that the actual propagation of the pulse takes up just a few lines of code. Most of the other code is simply plotting the results.

This example is contained in examples/simple_SSFM.py

.. code-block:: python
	
	import numpy as np
	import matplotlib.pyplot as plt
	import pynlo

	FWHM    = 0.050  # pulse duration (ps)
	pulseWL = 1550   # pulse central wavelength (nm)
	EPP     = 50e-12 # Energy per pulse (J)
	GDD     = 0.0    # Group delay dispersion (ps^2)
	TOD     = 0.0    # Third order dispersion (ps^3)

	Window  = 10.0   # simulation window (ps)
	Steps   = 100     # simulation steps
	Points  = 2**13  # simulation points

	beta2   = -120     # (ps^2/km)
	beta3   = 0.00     # (ps^3/km)
	beta4   = 0.005    # (ps^4/km)
        
	Length  = 20    # length in mm
    
	Alpha   = 0.0     # attentuation coefficient (dB/cm)
	Gamma   = 1000    # Gamma (1/(W km) 
    
	fibWL   = pulseWL # Center WL of fiber (nm)
    
	Raman   = True    # Enable Raman effect?
	Steep   = True    # Enable self steepening?

	alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m


	# set up plots for the results:
	fig = plt.figure(figsize=(8,8))
	ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
	ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
	ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
	ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)


	######## This is where the PyNLO magic happens! ############################

	# create the pulse!
	pulse = pynlo.light.DerivedPulses.SechPulse(power = 1, # Power will be scaled by set_epp
	                                            T0_ps                   = FWHM/1.76, 
	                                            center_wavelength_nm    = pulseWL, 
	                                            time_window_ps          = Window,
	                                            GDD=GDD, TOD=TOD, 
	                                            NPTS            = Points, 
	                                            frep_MHz        = 100, 
	                                            power_is_avg    = False)
	# set the pulse energy!
	pulse.set_epp(EPP) 

	# create the fiber!
	fiber1 = pynlo.media.fibers.fiber.FiberInstance()
	fiber1.generate_fiber(Length * 1e-3, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
	                              gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
                                
	# Propagation
	evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.005, USE_SIMPLE_RAMAN=True,
	                 disable_Raman              = np.logical_not(Raman), 
	                 disable_self_steepening    = np.logical_not(Steep))

	y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse, fiber=fiber1, n_steps=Steps)

	########## That's it! Physics complete. Just plotting commands from here! ################


	F = pulse.F_THz     # Frequency grid of pulse (THz)

	def dB(num):
	    return 10 * np.log10(np.abs(num)**2)
    
	zW = dB( np.transpose(AW)[:, (F > 0)] )
	zT = dB( np.transpose(AT) )

	y_mm = y * 1e3 # convert distance to mm

	ax0.plot(pulse_out.F_THz,    dB(pulse_out.AW),  color = 'r')
	ax1.plot(pulse_out.T_ps,     dB(pulse_out.AT),  color = 'r')

	ax0.plot(pulse.F_THz,    dB(pulse.AW),  color = 'b')
	ax1.plot(pulse.T_ps,     dB(pulse.AT),  color = 'b')

	extent = (np.min(F[F > 0]), np.max(F[F > 0]), 0, Length)
	ax2.imshow(zW, extent=extent, 
	           vmin=np.max(zW) - 40.0, vmax=np.max(zW), 
	           aspect='auto', origin='lower')

	extent = (np.min(pulse.T_ps), np.max(pulse.T_ps), np.min(y_mm), Length)
	ax3.imshow(zT, extent=extent, 
	           vmin=np.max(zT) - 40.0, vmax=np.max(zT), 
	           aspect='auto', origin='lower')
          

	ax0.set_ylabel('Intensity (dB)')
	ax0.set_ylim( - 80,  0)
	ax1.set_ylim( - 40, 40)

	ax2.set_ylabel('Propagation distance (mm)')
	ax2.set_xlabel('Frequency (THz)')
	ax2.set_xlim(0,400)

	ax3.set_xlabel('Time (ps)')

	plt.show()
	

Here are the results:

.. image:: https://cloud.githubusercontent.com/assets/1107796/14987706/d5dec8cc-110d-11e6-90eb-3cf14294b603.png
   :width: 500px
   :alt: results
   :align: center


Contributing
------------

We welcome suggestions for improvement, questions, comments, etc. The best way to to open a new issue here: https://github.com/pyNLO/PyNLO/issues/.


License
-------
PyNLO is licensed under the `GPLv3 license <http://choosealicense.com/licenses/gpl-3.0/>`_. This means that you are free to use PyNLO for any **open-source** project. Of course, PyNLO is provided "as is" with absolutely no warrenty.


References
----------
[1] Johan Hult, "A Fourth-Order Runge–Kutta in the Interaction Picture Method for Simulating Supercontinuum Generation in Optical Fibers," J. Lightwave Technol. 25, 3770-3775 (2007) https://www.osapublishing.org/jlt/abstract.cfm?uri=jlt-25-12-3770






