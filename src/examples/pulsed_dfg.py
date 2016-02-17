# -*- coding: utf-8 -*-
"""
Pulsed Difference Frequency Generation

This module sets up and then runs a difference frequency generation (DFG) 
simulation. The specific case is for two gaussian pulses at 1064 nm and 1550 nm
mixing in periodically poled lithium niobate (PPLN.)

"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pynlo.light.DerivedPulses import GaussianPulse
from pynlo.light import Pulse
from pynlo.media.crystals import PPLN
from pynlo.interactions.ThreeWaveMixing import dfg_problem
from pynlo.interactions.ThreeWaveMixing.multi_run_rw import DFGWriter

from pynlo.util import ode_solve
from pynlo.util.ode_solve import dopr853
from pynlo.light import OneDBeam

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pynlo.util.pynlo_ffts import IFFT_t

plt.close('all')


crystallength = 10.0e-3
crystal = PPLN(45, length = crystallength)

fr = 100.0
n_saves = 50

signal_wl = 1560.
signal_power = 0.100
signal_pulse_length = 0.150

pump_wl = 1048.
pump_power = 3.6
pump_waist = 50.0e-6
pump_pulse_length = 0.150


signal_in = GaussianPulse(power                 = signal_power,                          
                          T0_ps                 = signal_pulse_length,
                          center_wavelength_nm  = signal_wl,                          
                          time_window_ps        = 20.0, 
                          NPTS                  = 2**10, 
                          frep_MHz              = fr, 
                          power_is_avg          = True)
signal_in.add_time_offset(+0.250)
twind   = signal_in.time_window_ps
npoints = signal_in.NPTS

w0 = signal_in.calculate_weighted_avg_frequency_mks()*1.0e-12 * 2.0 * np.pi

# Use ideal quasi-phase matching period
crystal.set_pp(crystal.calculate_poling_period(pump_wl, signal_wl, None))


# output plots?
temporal = True
savefile = False
#
pump_in = GaussianPulse(power                   = pump_power,                          
                          T0_ps                 = pump_pulse_length,
                          center_wavelength_nm  = pump_wl,                          
                          time_window_ps        = 20.0, 
                          NPTS                  = 2**10, 
                          frep_MHz              = fr, 
                          power_is_avg          = True)
#

# Calculate peak optical intensity with class objects. Calculate the mean wavelength of
# each pulse, calculate the waist size (propagating from the crystal center out)
P0_p = np.max(np.abs(pump_in.AT)**2)
P0_s = np.max(np.abs(signal_in.AT)**2)

pump_beam = OneDBeam(pump_waist, this_pulse = pump_in)
sgnl_beam = OneDBeam(pump_waist, this_pulse = signal_in)

n_pump = pump_beam.get_n_in_crystal(pump_in, crystal)
n_sgnl = sgnl_beam.get_n_in_crystal(signal_in, crystal)

waist_p = pump_beam.calculate_waist(crystal.length_mks / 2.0,
                                    n_s = n_pump[int(len(n_pump)/2.0)])
waist_s = pump_beam.calculate_waist(crystal.length_mks / 2.0,
                                    n_s = n_pump[int(len(n_sgnl)/2.0)])

I_p = 2.0*P0_p/(np.pi * np.mean(waist_p)**2)
I_s = 2.0*P0_s/(np.pi * np.mean(waist_s)**2)

print 'peak intensity is ',np.sum((I_p+I_s))*1.0e-13,' GW/cm**2'


integrand = dfg_problem(pump_in, signal_in,  crystal,
              disable_SPM = True, pump_waist = pump_waist, plot_beam_overlaps = False)

# Set up integrator
rtol = 1.0e-6
atol = 1.0e-6
x0   = 0.0
x1   = crystallength
hmin = 0.0
h1   = 0.00001
out  = ode_solve.Output(n_saves)
beam_dim_fig = plt.figure()
a = ode_solve.ODEint(integrand.ystart, x0, x1, atol, rtol, h1,hmin, out,\
         dopr853.StepperDopr853, integrand)

a.integrate()

idler_in = integrand.idlr_in

res = integrand.process_stepper_output(a.out)
#                            
#print "pump power in: ",    pump_power_in, "mW"
#print "signal power in: ",  signal_power_in, "mW"
#print "idler power in: ",   idler_power_in, "mW"
#print "pump power out: ",   pump_power_out, "mW"
#print "signal power out: ", signal_power_out, "mW"
#print "idler power out: ",  idler_power_out, "mW"
#
#print "Total power before: ",np.sum(np.abs(pump_out[:,0])**2)+\
#                            np.sum(np.abs(sgnl_out[:,0])**2)+\
#                            np.sum(np.abs(idlr_out[:,0])**2)
#print "Total power after: ",np.sum(np.abs(pump_out[:,-1])**2)+\
#                            np.sum(np.abs(sgnl_out[:,-1])**2)+\
#                            np.sum(np.abs(idlr_out[:,-1])**2)

                         
if temporal:
    # temporal offset of outputs
    fig5 = plt.figure(figsize = (8,16))
    ip = []    
    for x in xrange(res.n_saves):
        p = res.get_idlr(x)
        ip.append(1e3 * p.calc_epp())
    #

    for x in xrange(res.n_saves):
        try:

            p = res.get_pump(x)
            ic = np.argmax(abs(p.AT)**2)
            tc = p.T_mks[ic]
            
            
            ax = plt.subplot(211)        

            p = res.get_pump(x)
            ax.plot((p.T_mks - tc)* 1e15,   abs(p.AT)**2 /
                     np.max(abs(res.pump_max_temporal)**2), 'b', label = 'pump', linewidth = 2)
            
            p = res.get_sgnl(x)
            ax.plot((p.T_mks - tc)* 1e15,   abs(p.AT)**2 /
                     np.max(abs(res.sgnl_max_temporal)**2), 'k', label = 'sgnl', linewidth = 2)
            
            p = res.get_idlr(x)
            ax.plot((p.T_mks - tc)* 1e15,   abs(p.AT)**2 /
                     np.max(abs(res.idlr_max_temporal)**2), 'r', label = 'idlr', linewidth = 2)
            
            plt.xlabel("Time (fs)")
            plt.ylabel("Normalized Power")
            plt.legend()
            plt.xlim(-3000*pump_pulse_length,3000*pump_pulse_length)
            plt.ylim(0, 1)
            ax = plt.subplot(212)
#            print '..'
            plt.plot(p.wl_mks*1.0e9, 
                     abs(p.AW)**2 / np.max(res.idlr_max_field**2),
                     label = 'idler', linewidth = 2)        
            plt.ylim(0, 1) 
            plt.xlim(2700, 3900)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Normalized Power (arb.)")        
            axins = inset_axes(ax, width = '20%',height = '10%', loc = 1)
 #           print '...'
            axins.plot(1000*np.array(res.zs[0:x]),ip[0:x] )
            axins.set_ylim(0, 1.1*max(ip))
            axins.set_xlim(res.zs[0], 1000.0*res.zs[-1])
            axins.set_xticks([])
            axins.set_yticks([])
            plt.draw()
            plt.savefig('c:\\frames\\frame%4.4d.png'%x)
            plt.clf()    
            if savefile:
                plt.savefig("time_overlap_exp.png", dpi = 300)
        except TypeError:
            print 'Failed to plot.'