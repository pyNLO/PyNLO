#!/usr/bin/env python

from distutils.core import setup

setup(name='pyNLO',
      version='0.1',
      description='Python nonlinear optics',
      author='Gabe Ycas',
      author_email='ycasg@colorado.edu',
      url='https://github.com/ycasg/PyNLO',
      packages=['pynlo'],
      package_dir = ['pynlo', 'src/pynlo']
     )