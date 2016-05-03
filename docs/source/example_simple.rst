Supercontinuum generation example
=================================

Here is an example of supercontinuum generation in a fiber ::

	import numpy as np
	import matplotlib.pyplot as plt
	import pynlo

	FWHM    = 0.050  # pulse duration (ps)
	pulseWL = 1550   # pulse central wavelength (nm)
	EPP     = 50e-12 # Energy per pulse (J)
	GDD     = 0.0    # Group delay dispersion (ps^2)
	TOD     = 0.0    # Third order dispersion (ps^3)

	Window  = 10.0   # simulation window (ps)
	Steps   = 50     # simulation steps
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
	fig = plt.figure(figsize=(10,10))
	ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
	ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
	ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
	ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)


	######## This is where the PyNLO magic happens! ############################

	# create the pulse!
	pulse = pynlo.light.DerivedPulses.SechPulse(1, FWHM/1.76, pulseWL, time_window_ps=Window,
	                  GDD=GDD, TOD=TOD, NPTS=Points, frep_MHz=100, power_is_avg=False)
	pulse.set_epp(EPP) # set the pulse energy 

	# create the fiber!
	fiber1 = pynlo.media.fibers.fiber.FiberInstance()
	fiber1.generate_fiber(Length * 1e-3, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
	                              gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
                                
	# Propagation
	evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=0.001, USE_SIMPLE_RAMAN=True,
	                 disable_Raman=np.logical_not(Raman), 
	                 disable_self_steepening=np.logical_not(Steep))

	y, AW, AT, pulse_out = evol.propagate(pulse_in=pulse, fiber=fiber1, n_steps=Steps)

	########## That's it! Physic done. Just boring plots from here! ################


	F = pulse.W_mks / (2 * np.pi) * 1e-12 # convert to THz

	def dB(num):
	    return 10 * np.log10(np.abs(num)**2)
    
	zW = dB( np.transpose(AW)[:, (F > 0)] )
	zT = dB( np.transpose(AT) )

	y = y * 1e3 # convert distance to mm


	ax0.plot(F[F > 0],  zW[-1], color='r')
	ax1.plot(pulse.T_ps,zT[-1], color='r')

	ax0.plot(F[F > 0],   zW[0], color='b')
	ax1.plot(pulse.T_ps, zT[0], color='b')


	extent = (np.min(F[F > 0]), np.max(F[F > 0]), 0, Length)
	ax2.imshow(zW, extent=extent, vmin=np.max(zW) - 60.0, 
	                 vmax=np.max(zW), aspect='auto', origin='lower')

	extent = (np.min(pulse.T_ps), np.max(pulse.T_ps), np.min(y), Length)
	ax3.imshow(zT, extent=extent, vmin=np.max(zT) - 60.0,
	           vmax=np.max(zT), aspect='auto', origin='lower')
          

	ax0.set_ylabel('Intensity (dB)')

	ax2.set_xlabel('Frequency (THz)')
	ax3.set_xlabel('Time (ps)')

	ax2.set_ylabel('Propagation distance (mm)')

	ax2.set_xlim(0,400)

	ax0.set_ylim(-80,0)
	ax1.set_ylim(-40,40)

	plt.show()
	

Output:

.. image:: https://cloud.githubusercontent.com/assets/1107796/13656636/c741acfc-e625-11e5-80e0-c291c17e2b2a.png
   :width: 500px
   :alt: example_result

	
