import numpy as np
import matplotlib.pyplot as plt
import pynlo
import matplotlib.cm as cm
import scipy.signal

pulseWL = 1550   # pulse central wavelength (nm)

GDD     = 0.0    # Group delay dispersion (ps^2)
TOD     = 0.0    # Third order dispersion (ps^3)

Window  = 10.0   # simulation window (ps)

beta2   = -40   # (ps^2/km)
beta3   = 0.00     # (ps^3/km)
beta4   = 0.001   # (ps^4/km)
        
Length  = 60    # length in mm
    
Alpha   = 0.0     # attentuation coefficient (dB/cm)
Gamma   = 1000    # Gamma (1/(W km) 
    
fibWL   = pulseWL # Center WL of fiber (nm)
    
Raman   = True    # Enable Raman effect?
Steep   = True    # Enable self steepening?

alpha = np.log((10**(Alpha * 0.1))) * 100  # convert from dB/cm to 1/m

# select a method for include noise on the input pulse:
noise_type = 'sqrt_N_freq'
# noise_type = 'one_photon_freq'


# DRAFT - use these parameters for a quick calculation
trials  = 2
error   = 0.1  # error desired by the integrator. Usually 0.001 is plenty good. Use larger values for speed
Steps   = 100     # simulation steps
Points  = 2**13  # simulation points

# FINAL - use these parameters for beautiful results that take a long time
# trials  = 5
# error   = 0.001  # error desired by the integrator. Usually 0.001 is plenty good. Use larger values for speed
# Steps   = 300     # simulation steps
# Points  = 2**12  # simulation points


# Coherent (use these parameters for a mostly-coherent pulse)
# FWHM    = 0.2  # pulse duration (ps)
# EPP     = 100e-12 # Energy per pulse (J)

# Not-so-coherent (use these parameters for a mostly-coherent pulse)
FWHM    = 0.35  # pulse duration (ps)
EPP     = 180e-12 # Energy per pulse (J)


# Incoherent (use these parameters for a mostly incoherent pulse)
# FWHM    = 0.8  # pulse duration (ps)
# EPP     = 800e-12 # Energy per pulse (J)


# set up plots for the results:
fig = plt.figure(figsize=(11,8))
ax0 = plt.subplot2grid((3,3), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,3), (1, 0), rowspan=2, sharex=ax0)

ax2 = plt.subplot2grid((3,3), (0, 1), rowspan=1)
ax3 = plt.subplot2grid((3,3), (1, 1), rowspan=2, sharex=ax2)

ax4 = plt.subplot2grid((3,3), (0, 2), rowspan=1)
ax5 = plt.subplot2grid((3,3), (1, 2), rowspan=2, sharex=ax4)

# create the fiber!
fiber1 = pynlo.media.fibers.fiber.FiberInstance()
fiber1.generate_fiber(Length * 1e-3, center_wl_nm=fibWL, betas=(beta2, beta3, beta4),
                              gamma_W_m=Gamma * 1e-3, gvd_units='ps^n/km', gain=-alpha)
                                
# Propagation
evol = pynlo.interactions.FourWaveMixing.SSFM.SSFM(local_error=error, USE_SIMPLE_RAMAN=True,
                 disable_Raman              = np.logical_not(Raman), 
                 disable_self_steepening    = np.logical_not(Steep))                                

# create the pulse!
pulse = pynlo.light.DerivedPulses.SechPulse(power = 1, # Power will be scaled by set_epp
                                            T0_ps                   = FWHM/1.76, 
                                            center_wavelength_nm    = pulseWL, 
                                            time_window_ps          = Window,
                                            GDD=GDD, TOD=TOD, 
                                            NPTS            = Points, 
                                            frep_MHz        = 100, 
                                            power_is_avg    = False)

pulse.set_epp(EPP) # set the pulse energy

g12, results = evol.calculate_coherence(pulse_in=pulse, fiber=fiber1, n_steps=Steps,
                                        num_trials=trials, noise_type=noise_type)

def dB(num):
    return 10 * np.log10(np.abs(num)**2)

for y, AW, AT, pulse_in, pulse_out in results:
    F = pulse_out.F_THz     # Frequency grid of pulse (THz)
    AW = AW.transpose()
    zW = dB(AW[:, (F > 0)] )
    ax0.plot(F[F>0],    zW[0],   color = 'b')
    ax0.plot(F[F>0],    zW[-1],  color = 'r')
    

    zT = dB( np.transpose(AT) )
    ax4.plot(pulse_out.T_ps,     dB(pulse_out.AT),  color = 'r')
    ax4.plot(pulse.T_ps,         dB(    pulse.AT),  color = 'b')


g12 = g12.transpose()

g12_line = g12[-1][F>0]

ax2.plot(F[F>0],g12_line, color='r')


extent = (np.min(F[F > 0]), np.max(F[F > 0]), 0, Length)
ax1.imshow(zW, extent=extent, 
           vmin=np.max(zW) - 40.0, vmax=np.max(zW), 
           aspect='auto', origin='lower')


extent = (np.min(F), np.max(F), 0, Length)
ax3.imshow(g12, extent=extent, clim=(0,1), aspect='auto', origin='lower', cmap=cm.inferno)

extent = (np.min(pulse.T_ps), np.max(pulse.T_ps), 0, Length)
ax5.imshow(zT, extent=extent, 
           vmin=np.max(zT) - 40.0, vmax=np.max(zT), 
           aspect='auto', origin='lower')
          

ax2.axhline(0, alpha=0.3, color='r')
ax2.axhline(1, alpha=0.3, color='g')

ax0.set_ylabel('Intensity (dB)')
ax2.set_ylabel('Coherence (g_12)')
ax4.set_ylabel('Intensity (dB)')

ax0.set_ylim( -60,  20)
ax2.set_ylim(-0.2, 1.2)
ax4.set_ylim( -20, 40)


ax1.set_ylabel('Propagation distance (mm)')
for ax in (ax1,ax3):
    ax.set_xlabel('Frequency (THz)')
    ax.set_xlim(0,400)


ax5.set_xlabel('Time (ps)')

title = '%i nm, %i fs, %i pJ'%(pulseWL, FWHM*1e3, EPP*1e12)

ax2.set_title(title, fontsize=16, weight='bold')
plt.tight_layout()

plt.savefig(title+'.png', dpi=200)

plt.show()