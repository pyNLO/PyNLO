# -*- coding: utf-8 -*-
"""
ODE solver, adapted from Numerical Recipes

@author: ycasg
"""
import numpy as np
import exceptions
import warnings

np.seterr(all='warn')

class Output:
    """ The output class is used by the ode solver to store the integrated output
    at specified *x* values. In addition to housing the matrices containing the
    *x* and *y* data, the class also provides a simple function call to store 
    new data and resizes the output grids dynamically.
    
    Parameters
    ----------
    nsaves
        Number of anticipated save points, used for calculating value of *x*
        at which integrand will be evaluted and saved. """
    kmax    = 0
    nvar    = 0
    nsave   = 0
    dense   = True
    count   = 0
    x1      = 0.0
    x2      = 0.0
    xout    = 0.0
    dxout   = 0.0
    xsave   = None
    ysave   = None
    def __init__(self, nsaves = None):

        if nsaves is None:
            self.kmax   = -1
            self.dense  = False
            self.count  = 0
        else:
            self.kmax   = 500
            self.nsave  = nsaves            
            self.count  = 0
            self.dense  = nsaves > 0
    def init(self, neqn, xlo, xhi, dtype = np.double):
        """ Setup routine, which creates the output arrays. If nsaves was provided
            at class initialization, the positions at which the integrand will be
            saved are also calculated.
            
            Parameters
            ----------
            neqn:
                Number of equations, or the number of y values at each x.
            xlo:
                Lower bound of integration (start point.)
            xhi:
                Upper bound of integration (stop point.)                
            dtype:
                Data type of each y. Any Python data type is acceptable.
            """
        self.nvar = neqn
        self.dtype = dtype
        if self.kmax == -1:
            return
        self.ysave = np.zeros((self.kmax, self.nvar), dtype = dtype)
        self.xsave = np.zeros((self.kmax), dtype = dtype)
        if self.dense:
            self.x1 = xlo
            self.x2 = xhi
            self.xout   = self.x1
            self.dxout  = (self.x2-self.x1) / float(self.nsave)
    def resize(self):
        kold = self.kmax
        self.kmax *= 2
        # Reshape x
        newx        = np.zeros((self.kmax,), dtype = self.xsave.dtype)
        newx[:kold] = self.xsave
        self.xsave  = newx
        # Reshape y
        newy        = np.zeros((self.kmax, self.nvar), dtype = self.dtype)
        newy[0:kold,:] = self.ysave
        self.ysave     = newy
    def save_dense(self, s, xout, h):
        if self.count == self.kmax:
            self.resize()
        y = s.dense_out(xout, h)
        if self.ysave.dtype != y.dtype:   
            errmsg = 'Integrand returns ',str(y.dtype),' but workspaces are initialized to ',str(self.ysave.dtype),'!'
            raise exceptions.TypeError(errmsg)
        self.ysave[self.count, :]   = y
        self.xsave[self.count]      = xout
        self.count += 1
    def save(self, x, y):
        if self.ysave.dtype != y.dtype:            
            errmsg = 'Integrand returns ',str(y.dtype),' but workspaces are initialized to ',str(self.ysave.dtype),'!'
            raise exceptions.TypeError(errmsg)
        if self.kmax <= 0:
            return
        if self.count == self.kmax:
            self.resize()
        self.ysave[self.count, :]   = y
        self.xsave[self.count]      = x
        self.count += 1        
    def out(self, nstp, x, y, s, h):
        """ nstp is current step number, current values are x & y, Stepper is s
        and step size is h"""
        if not self.dense:
            e = exceptions.AttributeError('Dense output is not set in Output!')
            raise e
        if nstp == 1:
            self.save(x,y)
            self.xout += self.dxout
        else:
            while (x-self.xout)*(self.x2-self.x1) > 0.0:
                self.save_dense(s, self.xout, h)
                self.xout += self.dxout
        
        

class StepperBase:
    x       = 0.0
    xold    = 0.0
    y       = None
    dydx    = None
    atol    = 0.0
    rtol    = 0.0
    dense   = True
    hdid    = 0.0
    hnext   = 0.0
    EPS     = np.finfo(np.double).eps
    n       = 0
    neqn    = 0
    yout    = None
    yerr    = None
    dtype   = None
    def __init__(self, yy, dydxx, xx, atoll, rtoll, dense):
        self.x = xx 
        self.y = yy # Reference assignments (y & yy are same object)
        self.dydx = dydxx
        self.atol = atoll
        self.rtol = rtoll
        self.dense = dense
        self.n = len(yy)
        self.neqn = self.n
        self.dtype = yy.dtype
        self.yout = self.gen_array()
        self.yerr = self.gen_array()
    def gen_array(self):
        return np.zeros( (self.n,), dtype = self.dtype)
        
class ODEint:
    MAXSTP  = 50000
    EPS     = np.finfo(np.double).eps
    nok     = 0
    nbad    = 0
    nvar    = 0
    x1       = 0.0
    x2      = 0.0
    hmin    = 0.0
    dense   = True     # True if dense output is requested
    y       = None
    dydx    = None
    ystart  = None
    output  = None
    RHS_class  = None    # function which evaluates dydx
    s       = None    # stepper class instance
    nstp    = 0    
    h       = 0.0
    def __init__(self, ystartt, xx1, xx2, atol, rtol, h1, hminn, outt,
                 stepper_class, RHS_class, dense = True, dtype = None):
         """ Class for integrating ODEs. 
         
         Notes
         -----
         This code is based upon *Numerical Recipes 3rd edition*'s 
         imlementation, but with some changes due to the translation:
         1.) The ODE is passed as a class instance 'RHS_class'. This class must
             have a member function deriv(x,y,dydx) which calculates the RHS
             and writes the value into dydx.
         2.) Unlike the NR version, ODEint is not derived from the stepper. 
             instead, the stepper class to be used is passed to the ODEint 
             constructor (stepper_class).
         3.) As a consequence of (2), x and y are stored in the stepper instance
             (ODEint.s) and not in ODEint iteself.  
         """
         self.nvar  = len(ystartt)         
         # If no dtype is specified, use the data type of ystartt
         if dtype is None:
             dtype = ystartt.dtype
         self.dtype = dtype
         self.y     = np.ndarray((self.nvar,), dtype=dtype)
         self.dydx  = np.ndarray((self.nvar,), dtype=dtype)
         self.ystart= ystartt         
         self.x1    = xx1
         self.x2    = xx2
         self.hmin  = hminn
         self.dense = dense
         self.out   = outt

         self.RHS_class= RHS_class
         self.s     = stepper_class(self.y, self.dydx, self.x1, atol, rtol, dense)         
         self.h     = np.abs(h1)*np.sign(xx2-xx1)
         self.y[:]  = self.ystart
         self.out.init(self.s.neqn, self.x1, self.x2, dtype = self.dtype)
    def integrate(self):
        self.RHS_class.deriv(self.x1, self.y, self.dydx)      
        if self.dense:
            self.out.out(-1, self.x1, self.y, self.s, self.h)
        else:
            self.out.save(self.x1, self.y)
        for self.nstp in xrange(self.MAXSTP):            
            if (self.s.x+self.h*1.0001 - self.x2 ) / (self.x2-self.x1) > 0.0:
                self.h = self.x2-self.s.x # If we would overshoot x2, reduce step size
            self.s.step(self.h, self.RHS_class)
            if self.s.hdid == self.h:
                self.nok+=1
            else:
                self.nbad+=1
            if self.dense:
                self.out.out(self.nstp, self.s.x, self.s.y, self.s, self.s.hdid)
            else:
                self.out.save(self.s.x, self.s.y)
            if (self.s.x-self.x2)*(self.x2-self.x1) >= 0:
                self.ystart[:] = self.s.y
                if self.out.kmax > 0 and\
                        abs(self.out.xsave[self.out.count-1]-self.x2) >\
                        100.0*abs(self.x2)*self.EPS:
                    self.out.save(self.s.x, self.s.y)
                return
            if abs(self.s.hnext) <= self.hmin:
                e = exceptions.RuntimeError('Step size below minimum specified value.')
                raise e
            self.h = self.s.hnext
        e = exceptions.RuntimeError('Integrator took too many steps without finishing.')
        raise e
