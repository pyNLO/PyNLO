# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 15:54:36 2014
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
"""
import numpy as np
import matplotlib.pyplot as plt

from pynlo.media.crystals.XTAL_PPLN import PPLN
from scipy import integrate, interpolate, signal

plt.close('all')

npoints = 2**6
crystallength = 25*1e-3
crystal = PPLN(45, length = crystallength)
#plt.style.use('seaborn-white')
plt.style.use('seaborn-talk')

def map_narrow1050(periods_bulk_m):
    """ Return WG periods for 15x16 um waveguide using NEL's calculation.
        Inputs and outputs assumed to be in meters"""
    xs = np.array([5.0,
                   22.56, 
                   23.80, 
                   26.20,
                   27.64,
                   28.70,
                   29.44,
                   29.96,
                   30.34,
                   30.46,
                   30.61,
                   47.0])
    ys = np.array([4.0,
                   21.58,
                   22.75,
                   24.97,
                   26.49,
                   27.54,
                   28.29,
                   28.84,
                   29.23,
                   29.36,
                   29.52,
                   46.0])
    return interpolate.interp1d(xs,ys)(periods_bulk_m*1e6)*1e-6
                   

def gen_dLdz_propto_BW(pump_wl = 1070., apodize = True):
    scale = 6.5e-9 # for propto BW
    def dLdz(L, z):
        signal = crystal.invert_dfg_qpm_to_signal_wl(pump_wl, L)
        bw = crystal.calculate_mix_phasematching_bw(pump_wl, signal)
        return (scale*bw)
    zs = np.linspace(0, 40, 4096+16) # Scale is and set to work at 40 mm crystal length
    Lambdas = integrate.odeint(dLdz, 23.75e-6, zs)    
    zs = np.linspace(0, 40, 4096) # Scale is and set to work at 40 mm crystal length
    Lambdas = np.reshape(Lambdas, len(Lambdas) )
    # Apodize
    if apodize:
        apod_offset = 5e-6
    else:
        apod_offset = 0
    Lambdas = np.hstack( (np.ones((4096,))*( Lambdas[0] ),
                          Lambdas,
                          np.ones((4096-16,))*( Lambdas[-1]+apod_offset )));
                             
    b,a = signal.bessel(2, 16/4096.)
    Lambdas = signal.filtfilt(b, a, Lambdas)[4096:4096+4096]        
    design = np.vstack( (zs * crystallength*1e3 / 40.0, Lambdas[::-1]) )
    return design
def gen_broadPump_narrow_Signal(signal_wl = 1550):
    # Pole from 25.4 um to 29 u
    #scale = 4.9e-9 # for propto BW
    scale = 1e-5 # for propto 1/BW
    #scale = (30 - 25.4)*1e-6/40.0
    def dLdz(L, z):
        pump = crystal.invert_dfg_qpm_to_pump_wl(signal_wl, L)
        bw = crystal.calculate_mix_phasematching_bw(pump, signal_wl)
        return (scale/bw)
        #return scale
    zs = np.linspace(0, 40,4096) # Scale is and set to work at 40 mm crystal length
    Lambdas = integrate.odeint(dLdz, 24.4e-6, zs)
    Lambdas = np.reshape(Lambdas, len(Lambdas) )
    # Apodize
    Lambdas = np.hstack( (np.ones((4096,))*( Lambdas[0]-5e-6 ),
                          Lambdas,
                          np.ones((4096,))*( Lambdas[-1]+5e-6 )));
                             
    b,a = signal.bessel(1, 16/4096.)
    Lambdas = signal.filtfilt(b, a, Lambdas)[4096:4096+4096]    
    design = np.vstack( (zs * crystallength*1e3 / 40.0, Lambdas[::-1]) )
    return design
def gen_linear_4to5um():
    # Pole from 28 to 32 um, for 2.8 to 4.0 um phase matching
    scale = (28-24)*1e-6/40.0
    def dLdz(L, z):
        return (scale)
    zs = np.linspace(0, 40,4096) # Scale is and set to work at 40 mm crystal length
    Lambdas = integrate.odeint(dLdz, 24e-6, zs)
    Lambdas = np.reshape(Lambdas, len(Lambdas) )
    # Apodize
    Lambdas = np.hstack( (np.ones((4096,))*( Lambdas[0]-5e-6 ),
                          Lambdas,
                          np.ones((4096,))*( Lambdas[-1]+5e-6 )));
                             
    b,a = signal.bessel(1, 16/4096.)
    Lambdas = signal.filtfilt(b, a, Lambdas)[4096:4096+4096]
    design = np.vstack( (zs * crystallength*1e3 / 40.0, Lambdas[::-1]) )
    return design

def gen_poled_regions(pump_wl = 1064., make_plots = False):
    # Poled regions, where the region length is fixed and the region spacing is
    # equal to the phase matching bandwidth. This should generate continuous
    # phase matching.
    transition_ratio = 0.4
    z = 0
    L = 33.5e-6 # perid to start at
    period_len = 2.15e-3
    design = [ [z+period_len/2, L] ]
    if make_plots:
        plt.figure()
    while L > 24.0e-6:        
        signal = crystal.invert_dfg_qpm_to_signal_wl(pump_wl, L)
        if make_plots:
            plt.plot(1e3*(z+period_len/2), 1/(1.0/pump_wl - 1.0/signal), 'ok')
        bw_invm_m = crystal.calculate_mix_phasematching_bw(pump_wl, signal)
        optical_bw = bw_invm_m / period_len 
        signal2 = 1.0e9/ ( 1/(signal*1e-9) + optical_bw)
        # Calculate the phase matching BW of the tentative signal, then use the
        # mean for calculating the change in poling period
        bw_invm_m_next = crystal.calculate_mix_phasematching_bw(pump_wl, signal2)
        optical_bw_next = bw_invm_m_next / period_len         
        z += period_len
        optical_bw_mean = 0.5*(optical_bw+optical_bw_next)
        print "bw mean", optical_bw_next
        signal2 = 1.0e9/ ( 1/(signal*1e-9) + optical_bw_mean)
        print "signal %f->%f"%(signal, signal2)
        L = crystal.calculate_poling_period(pump_wl, signal2, None)[0]
        print L
        design.append([z+period_len/2,L])
    print "**** Few Regions Poling Map *****"
    polingmap = np.array(design)
    print map_narrow1050(polingmap[:,1])*1e6
    print "**** Few Regions Poling Map *****"
    block_len = 1024
    zs = np.linspace(0, period_len/2.0, block_len)
    grating_zs = []
    grating_ps = []
    for x in range(len(polingmap)):
        p = polingmap[x]
        # First half of grating period
        zs_new = p[0] - zs[::-1]
        ps_new = np.ones((block_len,)) * p[1]
        if x != 0:
            transition_start = 0
            transition_end   = int(round(block_len*transition_ratio/2.0))
            transition_len   = transition_end-transition_start
            ps_ramp = polingmap[x-1,1] +0.5*(polingmap[x,1]-polingmap[x-1,1]) * (1+np.linspace(0, transition_len, transition_len) / float(transition_len))
            ps_new[transition_start:transition_end] = ps_ramp
        grating_zs.append(zs_new)
        grating_ps.append(ps_new)

        # Second half of grating period
        zs_new = p[0] + zs
        ps_new = np.ones((block_len,)) * p[1]
        if x != len(polingmap)-1:
            transition_start = block_len-int(round(block_len*transition_ratio/2.0))
            transition_end   = block_len
            transition_len   = transition_end-transition_start
            ps_ramp = polingmap[x,1] + 0.5*(polingmap[x+1,1]-polingmap[x,1]) * np.linspace(0, transition_len, transition_len) / float(transition_len)
            ps_new[transition_start:transition_end] = ps_ramp
        grating_zs.append(zs_new)
        grating_ps.append(ps_new)        
    if make_plots:        
        plt.xlabel("crystal position (mm)")
        plt.ylabel("idler wl (nm)")
        plt.show()
    grating_zs = np.array(grating_zs).flatten()
    grating_ps = np.array(grating_ps).flatten()        
    # Project onto output_zs. Note that the design grating length should be close
    # to crystallength for this to make sense
    if abs(crystallength - max(grating_zs))/crystallength > 0.1:
        print("""Warning: adjust design so that grating design length better 
                 matches crystal length. The current setting yields bad grating 
                 spacings, with a nominal crystal length of %d mm"""%( max(grating_zs*1e3)))
    output_zs = np.linspace(0, 1, 4096) * crystallength
    output_ps = interpolate.interp1d(grating_zs *max(output_zs)/ max(grating_zs), grating_ps)(output_zs)
    return np.vstack( (output_zs*1e3,output_ps) )

def gen_few_poled_regions(pump_wl = 1064., make_plots = False):
    # Poled regions, where the region length is fixed and the region spacing is
    # equal to the phase matching bandwidth. This should generate continuous
    # phase matching.
    transition_ratio = 0.2
    z = 0
    L = 32.0e-6 # perid to start at
    period_len = .75e-3
    design = [ [z+period_len/2, L] ]
    if make_plots:
        plt.figure()    
    while L > 24.5e-6:
        signal = crystal.invert_dfg_qpm_to_signal_wl(pump_wl, L)
        if make_plots:
            plt.plot(1e3*(z+period_len/2), 1/(1.0/pump_wl - 1.0/signal), 'ok')
        bw_invm_m = crystal.calculate_mix_phasematching_bw(pump_wl, signal)
        optical_bw = 0.5*bw_invm_m / period_len 
        signal2 = 1.0e9/ ( 1/(signal*1e-9) + optical_bw)
        # Calculate the phase matching BW of the tentative signal, then use the
        # mean for calculating the change in poling period
        bw_invm_m_next = crystal.calculate_mix_phasematching_bw(pump_wl, signal2)
        optical_bw_next = 0.5*bw_invm_m_next / period_len 
        print optical_bw_next
        z += period_len
        optical_bw_mean = 0.5*(optical_bw+optical_bw_next)
        signal2 = 1.0e9/ ( 1/(signal*1e-9) + optical_bw_mean)
        
        print "signal %f->%f"%(signal, signal2)
        L = crystal.calculate_poling_period(pump_wl, signal2, None)[0]
        print L
        design.append([z+period_len/2,L])
    polingmap = np.array(design)    
    print "**** Few Regions Poling Map *****"
    polingmap2 = np.array(design)
    polingmap2[:,0] *= 1e3
    polingmap2[:,1] *= 1e6
    print map_narrow1050(polingmap[:,1])*1e6
    print "**** Few Regions Poling Map *****"
    block_len = 1024
    zs = np.linspace(0, period_len/2.0, block_len)
    grating_zs = []
    grating_ps = []
    for x in range(len(polingmap)):
        p = polingmap[x]
        # First half of grating period
        zs_new = p[0] - zs[::-1]
        ps_new = np.ones((block_len,)) * p[1]
        if x != 0:
            transition_start = 0
            transition_end   = int(round(block_len*transition_ratio/2.0))
            transition_len   = transition_end-transition_start
            ps_ramp = polingmap[x-1,1] +0.5*(polingmap[x,1]-polingmap[x-1,1]) * (1+np.linspace(0, transition_len, transition_len) / float(transition_len))
            ps_new[transition_start:transition_end] = ps_ramp
        grating_zs.append(zs_new)
        grating_ps.append(ps_new)

        # Second half of grating period
        zs_new = p[0] + zs
        ps_new = np.ones((block_len,)) * p[1]
        if x != len(polingmap)-1:
            transition_start = block_len-int(round(block_len*transition_ratio/2.0))
            transition_end   = block_len
            transition_len   = transition_end-transition_start
            ps_ramp = polingmap[x,1] + 0.5*(polingmap[x+1,1]-polingmap[x,1]) * np.linspace(0, transition_len, transition_len) / float(transition_len)
            ps_new[transition_start:transition_end] = ps_ramp
        grating_zs.append(zs_new)
        grating_ps.append(ps_new)        
    if make_plots:        
        plt.xlabel("crystal position (mm)")
        plt.ylabel("idler wl (nm)")
        plt.show()

    grating_zs = np.array(grating_zs).flatten()
    grating_ps = np.array(grating_ps).flatten()        
    output_zs = np.linspace(0, 1, 4096) * crystallength
    output_ps = interpolate.interp1d(grating_zs *max(output_zs)/ max(grating_zs), grating_ps)(output_zs)
    return np.vstack( (output_zs*1e3,output_ps) )
    
plot_mode = "wavelengths"
detailed_plot = False
ctr = 1
titles = ["Flat BB (WG 1 and 6)", 
          "Broad 1 $\mu$m; also 3-4 $\mu$m (WG 2)", 
          "Broad 1 $\mu$m; also 3-4 $\mu$m (WG 2)", 
          "4-5$\mu$m (WG 3)", 
          "Many regions (WG 4)", 
          "Fewer regions (WG 5)"]
ref_wl = [1050, 1050, 1550, 1050, 1050, 1050]
ref_field = ["p","p","s","p","p","p"]
flat_bb_no_apod = lambda: gen_dLdz_propto_BW(apodize = False)
# Plot phase matching BW
#sgnls = np.linspace(1300,1600)
#pumps = np.linspace(980,1200)
#plt.subplot(121)
#plt.plot(sgnls, crystal.calculate_mix_phasematching_bw(1050, sgnls))
#plt.subplot(122)
#plt.plot(pumps, crystal.calculate_mix_phasematching_bw(pumps, 1550))


for x in [gen_dLdz_propto_BW, 
          gen_broadPump_narrow_Signal,
          gen_broadPump_narrow_Signal, 
          gen_linear_4to5um,
          gen_poled_regions,
          gen_few_poled_regions]:
    design = x()
    plt.subplot(2,3, ctr)
    if detailed_plot:
        plt.title(titles[ctr-1])
    if plot_mode == "periods":
        xs = design[0,:] - design[0,-1]/2.0
        fit = np.polyfit(xs ,map_narrow1050( design[1,:]),11)        
        if detailed_plot:
            plt.plot(design[0,:], map_narrow1050(design[1,:]) *1e6, label = 'WG 15x16')
            plt.plot(design[0,:], design[1,:] *1e6, label = 'bulk')
            plt.plot(design[0,:],np.poly1d(fit)(xs) *1e6, '--b', label = 'WG 15x16, fit', lw=2)
        else:
            plt.plot(design[0,:], map_narrow1050(design[1,:]) *1e6)
        plt.xlabel('Crystal position (mm)')
        plt.ylabel('$\Lambda$ (um)')
        
    if plot_mode == "wavelengths":
        for y in range(len(design[0,:]))[::10]:
            if ref_field[ctr-1] == "p":
                sgnl_wl = crystal.invert_dfg_qpm_to_signal_wl(ref_wl[ctr-1], design[1,y])
                idlr_wl = 1.0/( 1.0/ref_wl[ctr-1] - 1/sgnl_wl)                
                bw = 0.5 * crystal.calculate_mix_phasematching_bw(ref_wl[ctr-1], sgnl_wl)
            if ref_field[ctr-1] == "s":
                pump_wl = crystal.invert_dfg_qpm_to_pump_wl(ref_wl[ctr-1], design[1,y])
                idlr_wl = 1.0/( 1/pump_wl - 1.0/ref_wl[ctr-1])                                
                bw = 0.5 * crystal.calculate_mix_phasematching_bw(pump_wl, ref_wl[ctr-1])
            plt.plot(design[0,y], idlr_wl*1e-3 , '.b')
            pm_min = 1.0e6/ ( 1/(idlr_wl*1e-9) + bw/1e-3)
            pm_max = 1.0e6/ ( 1/(idlr_wl*1e-9) - bw/1e-3)
            plt.plot([design[0,y], design[0,y]], 
                     [pm_min, pm_max] , '-r', alpha = 0.2, lw=3)            
            
        plt.xlabel('Crystal position (mm)')        
        plt.ylabel('Color Phasematched ($\mu$m)')
        plt.yticks([2,3,4,5, 6])
    if ctr == 1:
        outdata_wg   = design[0,:]
        outdata_bulk = design[0,:]
        header = "Position from entrance [mm]"
    outdata_wg   = np.vstack( (outdata_wg,  map_narrow1050(design[1,:]) *1e6))
    outdata_bulk = np.vstack( (outdata_wg,                 design[1,:] *1e6))
    header = header + "," + titles[ctr-1]+ "[um]"
    if ctr==2 and detailed_plot:
        plt.legend(loc="best")

    ctr+=1    
np.savetxt(r"h:\wgdesigns_wgperiods.csv", outdata_wg.T, header = header, delimiter = ',', comments = "")
np.savetxt(r"h:\wgdesigns_bulkperiods.csv", outdata_bulk.T, header = header, delimiter = ',', comments = "")

plt.tight_layout()
plt.savefig("d:\\waveguide_designs_altview_talk.png", dpi = 600)
plt.show()