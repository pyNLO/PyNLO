#!/usr/bin/env python

from distutils.core import setup

setup(name='pyNLO',
      version='0.1',
      description='Python nonlinear optics',
      author='Gabe Ycas',
      author_email='ycasg@colorado.edu',
      url='https://github.com/ycasg/PyNLO',
      packages=['pynlo',
                'pynlo.devices',
                'pynlo.interactions',
                'pynlo.interactions.ThreeWaveMixing',
                'pynlo.interactions.FourWaveMixing',
                'pynlo.light',
                'pynlo.media',
                'pynlo.media.crystals',
                'pynlo.media.fibers',
                'pynlo.util',
                'pynlo.util.ode_solve'],
      package_dir = {'pynlo': 'src/pynlo'},
     )