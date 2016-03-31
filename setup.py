#!/usr/bin/env python

#from distutils.core import setup
# By popular demand...
from setuptools import setup
import os

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:
    import numpy as np
    install_requires=[
          "jsonpickle>=0.9.2",
          "jsonschema>=2.5.1",
          "mock>=1.0.1",
          "nose>=1.3.7",
          "nose-cov>=1.6",
          "nose-fixes>=1.3",
          "numpy>=1.9.2",          
          "scipy>=0.15.1"
          ]
else:
    np = None
    install_requires=[]


setup(name='pyNLO',
      version='0.1',
      description='Python nonlinear optics',
      author='Gabe Ycas',
      author_email='ycasg@colorado.edu',
      url='https://github.com/pyNLO/PyNLO',
      install_requires=install_requires,
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
      package_dir = {'': 'src'},
      package_data = {'pynlo': ['media/fibers/*.txt']},
    extras_require = {
        'DFG':  ["pyfftw"],
		},	  
     )