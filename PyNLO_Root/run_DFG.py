# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:54:36 2014

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pynlo.light.DerivedPulses import FROGPulse, SechPulse, NoisePulse
from pynlo.media.crystals import PPLN
from pynlo.interactions.ThreeWaveMixing import dfg_problem
from pynlo.util import ode_solve
from pynlo.util.ode_solve import dopr853

from gnlse_ffts import IFFT_t
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.close('all')

npoints = 2**9
crystallength = 3e-3
crystal = PPLN(45, length = crystallength)

fr = 100.0

n_saves = 500

pump_wl = 1040.
signal_wl = 1560.
pump_power = 2.0
signal_power = 0.100
twind = 10.
beamwaist = 40e-6

crystal.set_pp(crystal.calculate_poling_period(pump_wl, signal_wl, None))   
idler_wl = 1./(1./pump_wl - 1./signal_wl)

# output plots?
logplots = False
colormap = False
intidler = False
spectral = False
temporal = True
exptcomp = False
savefile = False

#pump_in.gen_frog(time_window = twind, center_wavelength = signal_wl, power = pump_power,
#                fileloc = 'broadened_er_pulse.dat',
#                flip_phase = True)

pump_pulse_length = 0.200
signal_pulse_length = 0.0500

#
pump_in = SechPulse(pump_power, pump_pulse_length, pump_wl, time_window = twind,
                    GDD = 0, TOD = 0.0, NPTS = npoints, frep_MHz = fr, power_is_avg = True)

#
signal_in = FROGPulse(center_wavelength_nm = signal_wl, power = signal_power,
                      time_window = twind, NPTS = npoints,
                      fileloc = '\\\\jake\\688\\OFM\\Maser\FROG\\frog_141020-  7\\Speck.dat',
                      flip_phase = True, frep_MHz=fr, power_is_avg = True)

integrand = dfg_problem(pump_in, signal_in, crystal,
              disable_SPM = True, waist = beamwaist)

# Set up integrator
rtol = 1.0e-6
atol = 1.0e-6
x0   = 0.0
x1   = crystallength
hmin = 0.0
h1   = 0.00001
out  = ode_solve.Output(n_saves)

a = ode_solve.ODEint(integrand.ystart, x0, x1, atol, rtol, h1,hmin, out,\
         dopr853.StepperDopr853, integrand)
a.integrate()

print 'integrated!'

pump_out = a.out.ysave[0:a.out.count, 0         : npoints].T
sgnl_out = a.out.ysave[0:a.out.count, npoints   :   2*npoints].T
idlr_out = a.out.ysave[0:a.out.count, 2*npoints :   3*npoints].T
z        = a.out.xsave[0:a.out.count]

pump_power_in =    np.round(1e3 * np.trapz(abs(IFFT_t(pump_out[:,0]))**2,
                            pump_in.T_mks) * pump_in.frep_Hz, decimals = 4)
signal_power_in =  np.round(1e3 * np.trapz(abs(IFFT_t(sgnl_out[:,0]))**2,
                            signal_in.T_mks) * signal_in.frep_Hz, decimals = 4)
pump_power_out =   np.round(1e3 * np.trapz(abs(IFFT_t(pump_out[:,-1]))**2,
                            pump_in.T_mks) * signal_in.frep_Hz, decimals = 4)
signal_power_out = np.round(1e3 * np.trapz(abs(IFFT_t(sgnl_out[:,-1]))**2,
                            signal_in.T_mks) * signal_in.frep_Hz, decimals = 4)
idler_power_out =  np.round(1e3 * np.trapz(abs(IFFT_t(idlr_out[:,-1]))**2,
                            integrand.idler.T_mks) * signal_in.frep_Hz, decimals = 4)
                            
print "pump power in: ",    pump_power_in, "mW"                           
print "signal power in: ",  signal_power_in, "mW"                           
print "pump power out: ",   pump_power_out, "mW"                           
print "signal power out: ", signal_power_out, "mW"                           
print "idler power out: ",  idler_power_out, "mW"    

# calculate each relevant FWHM
fwhm_pump_in = (pump_in.V_THz[abs(pump_out[:,0])**2 >
                max(abs(pump_out[:,0])**2) / 2] / (2 * np.pi))[-1] - \
               (pump_in.V_THz[abs(pump_out[:,0])**2 > 
                max(abs(pump_out[:,0])**2) / 2] / (2 * np.pi))[0]
               
fwhm_signal_in = (signal_in.V_THz[abs(sgnl_out[:,0])**2 > 
                  max(abs(sgnl_out[:,0])**2) / 2] / (2 * np.pi))[-1] - \
                 (signal_in.V_THz[abs(sgnl_out[:,0])**2 >
                  max(abs(sgnl_out[:,0])**2) / 2] / (2 * np.pi))[0]
                 
fwhm_idler_out = (integrand.idler_in.V_THz[abs(idlr_out[:,-1])**2 >
                  max(abs(idlr_out[:,-1])**2) / 2] / (2 * np.pi))[-1] - \
                 (integrand.idler_in.V_THz[abs(idlr_out[:,-1])**2 > 
                  max(abs(idlr_out[:,-1])**2) / 2] / (2 * np.pi))[0]

print "input pump spectral width:",   fwhm_pump_in, "THz"
print "input signal spectral width:", fwhm_signal_in, "THz"
print "output idler spectral width:", fwhm_idler_out, "THz"

if logplots:
    #log plots of all three output spectra
    fig1 = plt.figure()
    plt.semilogy(1/pump_in.wl*1e-2, abs(pump_out[:,10])**2, label = 'pump', linewidth = 2)
    plt.semilogy(1/signal_in.wl*1e-2, abs(sgnl_out[:,10])**2, label = 'signal', linewidth = 2)
    plt.semilogy(1/integrand.idler_in.wl*1e-2, abs(idlr_out[:,10])**2, label = 'idler', linewidth = 2)
    plt.xlabel(r"Wavenumber (cm$^{-1}$)")
    plt.ylabel("Spectral Power")
    plt.legend(loc = 2)
    plt.autoscale(tight = True)
    plt.ylim(ymin = 1e-3)
    if savefile:
        plt.savefig("logplot_exp.png", dpi = 300)
    plt.show()
        
if colormap:
    # color mesh of idler evolution
    fig2 = plt.figure()
    plt.pcolormesh(z*1e3, idler_in.wl*1e9, abs(idlr_out)**2)
    plt.autoscale(tight = True)
    plt.ylim(2000,4500)
    plt.xlabel("Crystal Length (mm)")
    plt.ylabel("Idler Wavelength (nm)")
    if savefile:
        plt.savefig("colormap_exp.png", dpi = 300)
    plt.show()
    
if intidler:
    # line plot of integrated idler evolution
    fig3 = plt.figure()
    plt.plot(z*1e3, sum(abs(idler_out),axis = 0), label = str(nsteps))
    plt.xlabel("Crystal Length (mm)")
    plt.ylabel("Integrated Idler Power")
    if savefile:
        plt.savefig("int_power_exp.png", dpi = 300)

if spectral:
    # spectral width of outputs as compared to center frequency
    fig4 = plt.figure()   
    plt.plot(pump_in.V / (2 * np.pi) / 1e12,
             abs(pump_out[:, 0])**2 / max(abs(pump_out[:, 0])**2),
             label = 'pump', linewidth = 2)
    plt.plot(signal_in.V / (2 * np.pi) / 1e12,
             abs(signal_out[:, 0])**2 / max(abs(signal_out[:, 0])**2),
             label = 'signal', linewidth = 2)
    plt.plot(idler_in.V / (2 * np.pi) / 1e12,
             abs(idler_out[:,-1])**2 / max(abs(idler_out[:, -1])**2),
             label = 'idler', linewidth = 2)
    plt.xlabel("Frequency from Center (THz)")
    plt.ylabel("Normalized Power (arb.)")
    plt.xlim(-25,25)
    plt.legend()
    if savefile:
        plt.savefig("spectral_width_exp.png", dpi = 300)
                                 
if temporal:
    # temporal offset of outputs
    fig5 = plt.figure(figsize = (8,16))
    ip = []    
    for x in xrange(len(z)):
        ip.append(1e3 * np.trapz(abs(IFFT_t(idlr_out[:,x]))**2,idler_in.T_mks) * signal_in.frep_Hz)
    #
    for x in xrange(0, len(pump_out[0,:])):
        ax = plt.subplot(211)
        ax.annotate('x = %.2f mm'%(1000.0*z[x]), xy=(3000*pump_pulse_length,0.7))

        ic = np.argmax(abs(IFFT_t(pump_out[:,  x]))**2)
        tc = pump_in.T_mks[ic]
        ax.plot((pump_in.T_mks - tc)* 1e15,   abs(IFFT_t(pump_out[:,  x]))**2 /
                 np.max(abs(IFFT_t(pump_out[:,  :]))**2), 'b', label = 'pump', linewidth = 2)
        ax.plot((signal_in.T_mks-tc) * 1e15, abs(IFFT_t(sgnl_out[:,x]))**2 /
                 np.max(abs(IFFT_t(sgnl_out[:,  :]))**2), 'k',  label = 'signal', linewidth = 2)
        ax.plot((idler_in.T_mks-tc) * 1e15,  abs(IFFT_t(idlr_out[:, x]))**2 /
                 np.max(abs(IFFT_t(idlr_out[:,  :]))**2), 'r', label = 'idler', linewidth = 2)
        plt.xlabel("Time (fs)")
        plt.ylabel("Normalized Power")
        plt.legend()
        plt.xlim(-3000*pump_pulse_length,3000*pump_pulse_length)
        plt.ylim(0, 1)
        ax = plt.subplot(212)
        plt.plot(idler_in.wl_mks*1.0e6, 
                 abs(idlr_out[:,x])**2 / np.max(abs(idlr_out[:, :])**2),
                 label = 'idler', linewidth = 2)        
        plt.ylim(0, 1) 
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Normalized Power (arb.)")        
        axins = inset_axes(ax, width = '20%',height = '10%', loc = 1)
        axins.plot(1000*np.array(z[0:x]), ip[0:x])
        axins.set_ylim(0, 1.1*max(ip))
        axins.set_xlim(z[0], 1000.0*z[-1])
        axins.set_xticks([])
        axins.set_yticks([])
        plt.draw()
        plt.savefig('e:\\frames\\frame%4.4d.png'%x)
        plt.clf()

    if savefile:
        plt.savefig("time_overlap_exp.png", dpi = 300)

if exptcomp:
    # compare normalized idler output to measured normalized output from Flavio
    data = np.genfromtxt("O:\\OFM\\Maser\\Dual-Comb 100 MHz System\\MIR spectra-101714.csv",
                         delimiter = ',')
    fig6 = plt.figure()
    plt.plot(idler_in.wl * 1e6, abs(idler_out[:, -1])**2 / max(abs(idler_out[:, -1])**2),
             label = 'simulated', linewidth = 2)
    plt.plot(data[:,0] * 1e6, data[:,1] / max(data[:,1]), '--', label = 'experimental',
             linewidth = 2)
    plt.legend()
    plt.xlim(2.6, 3.6)
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Spectral Power (normalized)")
    if savefile:
        plt.savefig("exp_data_compare_08um.png", dpi = 300)