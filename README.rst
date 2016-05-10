pyNLO: Nonlinear optics modeling for Python
===========================================

This README is best viewed at http://pynlo.readthedocs.io/en/latest/readme_link.html

Complete documentation is available at http://pynlo.readthedocs.io/

.. image:: https://cloud.githubusercontent.com/assets/1107796/13850062/17f09ea8-ec1e-11e5-9311-b94df29c01cb.png
   :width: 330px
   :alt: PyNLO
   :align: right


Introduction
------------

PyNLO provides an easy-to-use, object-oriented set of tools for modeling the nonlinear interaction of light with materials. It provides many functionalities for representing pulses of light, beams of light, and nonlinear materials, such as crystals and fibers. Also, it features methods for simulating both three-wave-mixing processes (such as DFG), as well as four-wave-mixing processes such as supercontinuum generation. 

Features:
	- A solver for the propagation of light through a Chi-3 material, useful for simulation pulse compression and supercontinuum generation in an optical fiber. This solver is highly efficient, thanks to an adaptive-step-size implementation of the "Fourth-order Runge-Kutta in the Interaction Picture " (RK4IP) method of `Hult (2007) <https://www.osapublishing.org/jlt/abstract.cfm?uri=jlt-25-12-3770>`_.
	
	- A solver for difference frequency generation. 
	
	- ...and much more!


Installation
------------

PyNLO requires Python 2.7 or 3.3-3.5. If you don't already have Python, we recommend an "all in one" Python package such as the `Anaconda Python Distribution <https://www.continuum.io/downloads>`_, which is available for free.

With pip
~~~~~~~~

The latest release can be installed from PyPi with ::

    pip install pynlo

With setuptools
~~~~~~~~~~~~~~~

If you prefer the development version from GitHub, download it here, `cd` to the PyNLO directory, and use ::

    python setup.py install

Or, if you wish to edit the PyNLO source code without re-installing each time ::

    python setup.py develop


Example of use
--------------

Simple example goes here:

.. code-block:: python

	import pynlo
	
	pynlo......
	

Documentation
-------------
The complete documentation for PyNLO is availabe at https://pynlo.readthedocs.org.


Contributing
------------

We welcome suggestions for improvement! The best way to to open a new issue here: https://github.com/pyNLO/PyNLO/issues/.


License
-------
PyNLO is licensed under the `GPLv3 license <http://choosealicense.com/licenses/gpl-3.0/>`_. This means that you are free to use PyNLO for any **open-source** project. Of course, PyNLO is provided "as is" with absolutely no warrenty.


References
----------
[1] Johan Hult, "A Fourth-Order Runge–Kutta in the Interaction Picture Method for Simulating Supercontinuum Generation in Optical Fibers," J. Lightwave Technol. 25, 3770-3775 (2007) https://www.osapublishing.org/jlt/abstract.cfm?uri=jlt-25-12-3770






