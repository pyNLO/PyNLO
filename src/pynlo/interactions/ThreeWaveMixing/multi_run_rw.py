# -*- coding: utf-8 -*-
"""
DFG integation results writer. Saves numerically integrated
"DFGintegrand" using pyTables / HDF5.
"""
import numpy as np
import tables
from pynlo.util.pynlo_ffts import IFFT_t
from pynlo.light.PulseBase import Pulse
import exceptions

class DFGReader:
    """ Class to read saved DFG modeling runs."""
    run_ctr     = 0
    int_ctr     = 0
    root_name   = ""
    root        = None
    tables_file = None
    
    def __init__(self, file_name, param_name):
        """ Initialize by opening HDF5 file with results for reading. """
        self.tables_file = tables.open_file(file_name, mode = 'r')
        self.root = self.tables_file.get_node('/' + param_name)
        self.root_name = '/' + param_name        
        self.run_ctr = 0
        for n in self.tables_file.list_nodes(self.root_name):
            self.run_ctr += 1
        self.frep_Hz    = self.tables_file.get_node(self.root_name+'/run0/frep_Hz')[0]        


    def get_run_param(self, run_num = None):
        """ Return value of changing paramter for run 'run_num'. By default, 
            returns value for last run."""
        if run_num is None:
            run_num = self.run_ctr - 1
        if run_num < 0 or run_num >= self.run_ctr:
            raise exceptions.IndexError(str(run_num)+' not a valid run index.')
        param = self.tables_file.get_node(self.root_name+'/run'+str(run_num)+\
                '/param')[0]
        return param
        
    def get_run(self, run_num = None, z_num = None):
        """ Return tuple of (pump, signal, idler) pulses for run # n at z index
        z_num . By default, returns last z step of last run."""
        if run_num is None:
            run_num = self.run_ctr - 1
        if run_num < 0 or run_num >= self.run_ctr:
            raise exceptions.IndexError(str(run_num)+' not a valid run index.')
        if z_num is None:
            z_num = -1
            
        run = self.root_name+'/run'+str(run_num)
        
        # Read in and generate pump pulse
        pump_field  = self.tables_file.get_node(run+'/pump')[:,z_num]
        pump_Twind  = self.tables_file.get_node(run+'/pump_Twind')[0]
        pump_center = self.tables_file.get_node(run+'/pump_center_nm')[0]
        p = Pulse()
        p.set_NPTS(len(pump_field))
        p.set_center_wavelength_nm(pump_center)
        p.set_time_window_ps(pump_Twind)
        p.set_AW(pump_field)
        p.set_frep_MHz(self.frep_Hz * 1.0e-6)
        
        # Read in and generate signal pulse
        sgnl_field  = self.tables_file.get_node(run+'/sgnl')[:,z_num]
        sgnl_Twind  = self.tables_file.get_node(run+'/sgnl_Twind')[0]
        sgnl_center = self.tables_file.get_node(run+'/sgnl_center_nm')[0]
        s = Pulse()
        s.set_NPTS(len(sgnl_field))
        s.set_center_wavelength_nm(sgnl_center)
        s.set_time_window_ps(sgnl_Twind)
        s.set_AW(sgnl_field)
        s.set_frep_MHz(self.frep_Hz * 1.0e-6)

        # Read in and generate idler pulse
        idlr_field  = self.tables_file.get_node(run+'/idlr')[:,z_num]
        idlr_Twind  = self.tables_file.get_node(run+'/idlr_Twind')[0]
        idlr_center = self.tables_file.get_node(run+'/idlr_center_nm')[0]
        i = Pulse()
        i.set_NPTS(len(idlr_field))
        i.set_center_wavelength_nm(idlr_center)
        i.set_time_window_ps(idlr_Twind)
        i.set_AW(idlr_field)
        i.set_frep_MHz(self.frep_Hz * 1.0e-6)
        
        return (p, s, i)
        
    def get_next_run(self, z_num = None) :
        if self.int_ctr < self.run_ctr:
            self.int_ctr += 1
            return self.get_run(self.int_ctr - 1, z_num)

class DFGWriter:
    run_ctr     = 0    
    def __init__(self, file_name, param_name):
        self.tables_file = tables.open_file(file_name, mode = 'a')
        try:
            self.root = self.tables_file.create_group('/', param_name)
        except tables.NodeError:
            print 'Parameter "'+param_name+'" already in table; deleting existing data.'
            self.tables_file.remove_node('/', param_name)
            self.root = self.tables_file.create_group('/', param_name)
            
    def add_run(self,integrand, odesolver, param):
        grp = self.tables_file.create_group(self.root, 'run'+str(self.run_ctr), title=str(param))
        npts           = odesolver.nvar / 3
        count          = odesolver.out.nsave
        
        # Store pump, signal, idler fields        
        atom = tables.Atom.from_dtype(odesolver.out.ysave[0:count, 0 : npts].T.dtype)
        data_shape = odesolver.out.ysave[0:count, 0 : npts].T.shape
        pump_data = self.tables_file.createCArray(grp, 'pump', atom, data_shape)
        pump_data[:] = odesolver.out.ysave[0:count, 0 : npts].T
        sgnl_data = self.tables_file.createCArray(grp, 'sgnl', atom, data_shape)      
        sgnl_data[:] = odesolver.out.ysave[0:count, npts : 2 * npts].T
        idlr_data = self.tables_file.createCArray(grp, 'idlr', atom, data_shape)
        idlr_data[:] = odesolver.out.ysave[0:count, 2*npts : 3 * npts].T
        
        # By the design of the pulse class, the three things we need are center
        # wavelength, time window, and rep rate (N points is set by the array sizes)
        atom = tables.Atom.from_dtype(np.dtype(np.double))
        data_shape = (1,)
        pump_twind = self.tables_file.createCArray(grp, 'pump_Twind', atom, data_shape)
        pump_twind[:] = integrand.pump.time_window_ps
        sgnl_twind = self.tables_file.createCArray(grp, 'sgnl_Twind', atom, data_shape)
        sgnl_twind[:] = integrand.sgnl.time_window_ps
        idlr_twind = self.tables_file.createCArray(grp, 'idlr_Twind', atom, data_shape)
        idlr_twind[:] = integrand.idlr.time_window_ps
        
        atom = tables.Atom.from_dtype(np.dtype(np.double))
        data_shape = (1,)
        pump_wlc = self.tables_file.createCArray(grp, 'pump_center_nm', atom, data_shape)
        pump_wlc[:] = integrand.pump.center_wavelength_nm
        sgnl_wlc = self.tables_file.createCArray(grp, 'sgnl_center_nm', atom, data_shape)
        sgnl_wlc[:] = integrand.sgnl.center_wavelength_nm
        idlr_wlc = self.tables_file.createCArray(grp, 'idlr_center_nm', atom, data_shape)
        idlr_wlc[:] = integrand.idlr.center_wavelength_nm
        
        atom = tables.Atom.from_dtype(np.dtype(np.double))
        data_shape = (1,)
        fr = self.tables_file.createCArray(grp, 'frep_Hz', atom, data_shape)
        fr[:] = integrand.pump.frep_Hz
        
        atom = tables.Atom.from_dtype(np.dtype(np.double))
        data_shape = (1,)
        para = self.tables_file.createCArray(grp, 'param', atom, data_shape)
        para[:] = param
        
        self.tables_file.flush()
        self.run_ctr += 1
