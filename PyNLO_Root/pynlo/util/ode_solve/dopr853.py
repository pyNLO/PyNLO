# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 17:21:09 2015
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
@author: ycasg
"""

from pynlo.util.ode_solve.steppers import StepperBase
import numpy as np
from pynlo.util.ode_solve import dopr853_constants as dc
from pynlo.util.ode_solve.dopr853_controller import Controller
import exceptions

class StepperDopr853(StepperBase):
    dtype = None
    yerr2 = None
    k2= None;k3= None;k4= None; k5= None;k6= None;k7= None;k8= None;k9= None;k10 = None
    rcont1= None;rcont2= None;rcont3= None;rcont4= None;rcont5= None;rcont6= None
    rcont7= None;rcont8 = None
    con = Controller()
    def __init__(self, yy, dydxx, xx, atoll, rtoll, dens):
        StepperBase.__init__(self, yy,dydxx, xx, atoll, rtoll, dens)
        self.yerr2  = self.gen_array()
        self.k2     = self.gen_array()
        self.k3     = self.gen_array()
        self.k4     = self.gen_array()
        self.k5     = self.gen_array()
        self.k6     = self.gen_array()
        self.k7     = self.gen_array()
        self.k8     = self.gen_array()
        self.k9     = self.gen_array()
        self.k10    = self.gen_array()
        self.rcont1 = self.gen_array()
        self.rcont2 = self.gen_array()
        self.rcont3 = self.gen_array()
        self.rcont4 = self.gen_array()
        self.rcont5 = self.gen_array()
        self.rcont6 = self.gen_array()
        self.rcont7 = self.gen_array()
        self.rcont8 = self.gen_array()
        self.EPS    = np.finfo(np.double).eps
    def step(self, htry, RHS_class):
        h = htry
        dydxnew  = self.gen_array()
        while True:
            self.dy(h, RHS_class)            
            err = self.error(h)
            success, h = self.con.success(err, h)
            if success:
                break
            if abs(h) <= abs(self.x)*self.EPS:
                e = exceptions.OverflowError('stepsize underflow in StepperDopri853')
                raise e
        RHS_class.deriv(self.x + h, self.yout, dydxnew)
        if self.dense:
            self.prepare_dense(h, dydxnew, RHS_class)
        self.dydx[:]= dydxnew
        self.y      = self.yout
        self.xold   = self.x
        self.hdid   = h
        self.x      += self.hdid
        self.hnext  = self.con.hnext
    def dy(self, h, RHS_class):
        # dy estimator. Like RK5, but more -- 12 stages!
        ytemp = self.gen_array()
        y       = self.y
        dydx    = self.dydx
        x       = self.x
        if not (y is self.y and dydx is self.dydx and x is self.x):
            raise exceptions.AssertionError('Oh noes!')
        # 1
        ytemp[:]=self.y+h*dc.a21*self.dydx[:]
        # 2
        RHS_class.deriv(x+dc.c2*h,ytemp,self.k2)
        ytemp[:]=y+h*(dc.a31*dydx+dc.a32*self.k2)
        # 3
        RHS_class.deriv(x+dc.c3*h,ytemp,self.k3)
        ytemp[:]=y+h*(dc.a41*dydx+dc.a43*self.k3)
        # 4
        RHS_class.deriv(x+dc.c4*h,ytemp,self.k4)
        ytemp[:]=y+h*(dc.a51*dydx+dc.a53*self.k3+dc.a54*self.k4)
        # 5
        RHS_class.deriv(x+dc.c5*h,ytemp,self.k5)
        ytemp[:]=y+h*(dc.a61*dydx+dc.a64*self.k4+dc.a65*self.k5)
        # 6
        RHS_class.deriv(x+dc.c6*h,ytemp,self.k6)	
        ytemp[:]=y+h*(dc.a71*dydx+dc.a74*self.k4+dc.a75*self.k5+dc.a76*self.k6)
        # 7
        RHS_class.deriv(x+dc.c7*h,ytemp,self.k7)
        ytemp[:]=y+h*(dc.a81*dydx+dc.a84*self.k4+dc.a85*self.k5+dc.a86*self.k6+dc.a87*self.k7)        
        # 8
        RHS_class.deriv(x+dc.c8*h,ytemp,self.k8)
        ytemp[:]=y+h*(dc.a91*dydx+dc.a94*self.k4+dc.a95*self.k5+dc.a96*self.k6+dc.a97*self.k7+dc.a98*self.k8)
        # 9
        RHS_class.deriv(x+dc.c9*h,ytemp,self.k9)
        ytemp[:]=y+h*(dc.a101*dydx+dc.a104*self.k4+dc.a105*self.k5+dc.a106*self.k6+dc.a107*self.k7+dc.a108*self.k8+dc.a109*self.k9)
        # 10
        RHS_class.deriv(x+dc.c10*h,ytemp,self.k10)
        ytemp[:]=y+h*(dc.a111*dydx+dc.a114*self.k4+dc.a115*self.k5+dc.a116*self.k6+dc.a117*self.k7+dc.a118*self.k8+dc.a119*self.k9+dc.a1110*self.k10)
        # 11
        RHS_class.deriv(x+dc.c11*h,ytemp,self.k2)
        xph=x+h
        ytemp[:]=y+h*(dc.a121*dydx+dc.a124*self.k4+dc.a125*self.k5+dc.a126*self.k6+dc.a127*self.k7+dc.a128*self.k8+dc.a129*self.k9+dc.a1210*self.k10+dc.a1211*self.k2);
        # 12
        RHS_class.deriv(xph,ytemp,self.k3)
        self.k4[:]=dc.b1*dydx+dc.b6*self.k6+dc.b7*self.k7+dc.b8*self.k8+dc.b9*self.k9+dc.b10*self.k10+dc.b11*self.k2+dc.b12*self.k3
        # yout:
        self.yout=y+h*self.k4
        # Two error estimators:
        self.yerr[:]=self.k4-dc.bhh1*dydx-dc.bhh2*self.k9-dc.bhh3*self.k3;
        self.yerr2[:]=dc.er1*dydx+dc.er6*self.k6+dc.er7*self.k7+dc.er8*self.k8+dc.er9*self.k9+dc.er10*self.k10+dc.er11*self.k2+dc.er12*self.k3
    def prepare_dense(self, h,dydxnew, RHS_class):
        ydiff = self.gen_array()
        bspl  = self.gen_array()
        ytemp = self.gen_array()
        
        self.rcont1[:]=self.y
        ydiff=self.yout-self.y
        self.rcont2[:]=ydiff;
        bspl=h*self.dydx-ydiff;
        self.rcont3[:]=bspl;
        self.rcont4[:]=ydiff-h*dydxnew-bspl
        self.rcont5[:]=dc.d41*self.dydx+dc.d46*self.k6+dc.d47*self.k7+dc.d48*self.k8+\
            dc.d49*self.k9+dc.d410*self.k10+dc.d411*self.k2+dc.d412*self.k3
        self.rcont6[:]=dc.d51*self.dydx+dc.d56*self.k6+dc.d57*self.k7+dc.d58*self.k8+\
            dc.d59*self.k9+dc.d510*self.k10+dc.d511*self.k2+dc.d512*self.k3
        self.rcont7[:]=dc.d61*self.dydx+dc.d66*self.k6+dc.d67*self.k7+dc.d68*self.k8+\
            dc.d69*self.k9+dc.d610*self.k10+dc.d611*self.k2+dc.d612*self.k3
        self.rcont8[:]=dc.d71*self.dydx+dc.d76*self.k6+dc.d77*self.k7+dc.d78*self.k8+\
            dc.d79*self.k9+dc.d710*self.k10+dc.d711*self.k2+dc.d712*self.k3
        ytemp[:]=self.y+h*(dc.a141*self.dydx+dc.a147*self.k7+dc.a148*self.k8+dc.a149*self.k9+\
            dc.a1410*self.k10+dc.a1411*self.k2+dc.a1412*self.k3+dc.a1413*dydxnew);
        RHS_class.deriv(self.x+dc.c14*h,ytemp,self.k10);
        ytemp[:]=self.y+h*(dc.a151*self.dydx+dc.a156*self.k6+dc.a157*self.k7+dc.a158*self.k8+\
            dc.a1511*self.k2+dc.a1512*self.k3+dc.a1513*dydxnew+dc.a1514*self.k10)
        RHS_class.deriv(self.x+dc.c15*h,ytemp,self.k2)
        ytemp[:]=self.y+h*(dc.a161*self.dydx+dc.a166*self.k6+dc.a167*self.k7+dc.a168*self.k8+\
            dc.a169*self.k9+dc.a1613*dydxnew+dc.a1614*self.k10+dc.a1615*self.k2)
        RHS_class.deriv(self.x+dc.c16*h,ytemp,self.k3)
        self.rcont5[:]=h*(self.rcont5+dc.d413*dydxnew+dc.d414*self.k10+dc.d415*self.k2+dc.d416*self.k3)
        self.rcont6[:]=h*(self.rcont6+dc.d513*dydxnew+dc.d514*self.k10+dc.d515*self.k2+dc.d516*self.k3)
        self.rcont7[:]=h*(self.rcont7+dc.d613*dydxnew+dc.d614*self.k10+dc.d615*self.k2+dc.d616*self.k3)
        self.rcont8[:]=h*(self.rcont8+dc.d713*dydxnew+dc.d714*self.k10+dc.d715*self.k2+dc.d716*self.k3)
    def dense_out(self, x, h):
        s = (x - self.xold) / h
        s1 = 1.0 - s
        return self.rcont1+s*(self.rcont2+s1*(self.rcont3+s*(self.rcont4+s1*(self.rcont5+\
		s*(self.rcont6+s1*(self.rcont7+s*self.rcont8))))))
    def error(self, h):
        err1 = 0.0
        err2 = 0.0
        sk   = self.gen_array()
        # numpy.maximum checks two arrays element-wise and returns larger value for each
        sk[:] = self.atol + self.rtol*np.maximum(np.abs(self.y), np.abs(self.yout))
        err2 = np.sum( pow(np.abs(self.yerr )/sk, 2) )
        err1 = np.sum( pow(np.abs(self.yerr2)/sk, 2) )
        deno = err1 + 0.01 * err2
        if deno < 0.0:
            deno = 1.0
        return abs(abs(h) * np.abs(err1) * np.sqrt(1.0 / (deno*self.n)))
        