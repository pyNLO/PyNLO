PyNLO package
=============



pynlo.light
-----------

The **light** module contains modules to model light pulses. 


pynlo.light.PulseBase
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pynlo.light.PulseBase.Pulse
	:members:
	:special-members:
	:show-inheritance:
	
pynlo.light.DerivedPulses
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pynlo.light.DerivedPulses.SechPulse
	:members:
	:special-members:

.. autoclass:: pynlo.light.DerivedPulses.GaussianPulse
	:members:
	:special-members:
	:show-inheritance:

.. autoclass:: pynlo.light.DerivedPulses.FROGPulse
	:members:
	:special-members:
	:show-inheritance:

.. autoclass:: pynlo.light.DerivedPulses.NoisePulse
	:members:
	:special-members:
	:show-inheritance:

.. autoclass:: pynlo.light.DerivedPulses.CWPulse
	:members:
	:special-members:
	:show-inheritance:
	
pynlo.light.beam
~~~~~~~~~~~~~~~~

.. autoclass:: pynlo.light.beam.OneDBeam
	:members:
	:special-members:
	:show-inheritance:
	


pynlo.interactions
------------------

The pynlo.interactions module contains sub-modules to simulate the interaction in both three-wave-mixing (like DFG) and four-wave mixing (like supercontinuum generation).


pynlo.interactions.FourWaveMixing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements the Split-step Fourier Method to solve the Generalized Nonlinear Schrodiner Equation and simulate the propagation of pulses in a Chi-3 nonlinear medium.

.. autoclass:: pynlo.interactions.FourWaveMixing.SSFM.SSFM
	:members: __init__, propagate, propagate_to_gain_goal, calculate_coherence
	:show-inheritance:
	

pynlo.interactions.ThreeWaveMixing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module simulated DFG in a Chi-2 medium.

.. autoclass:: pynlo.interactions.ThreeWaveMixing.DFG_integrand.dfg_problem
    :members:
    :undoc-members:
	:special-members:
    :show-inheritance:
	
.. autoclass:: pynlo.interactions.ThreeWaveMixing.DFG.NLmix
	    :members:
	    :undoc-members:
		:special-members:
	    :show-inheritance:
		


pynlo.media
-----------

The **media** module contains sub-modules for modeling fibers and crystals.


pynlo.media.fibers
~~~~~~~~~~~~~~~~~~

These classes are used to model fibers or fiber-like waveguides.

.. autoclass:: pynlo.media.fibers.fiber.FiberInstance
    :members:
    :undoc-members:
	:special-members:
    :show-inheritance:

.. automodule:: pynlo.media.fibers.calculators
    :members:
    :show-inheritance:


pynlo.crystals
~~~~~~~~~~~~~~

These classes are used to model various nonlinear crystals.

.. autoclass:: pynlo.media.crystals.CrystalContainer.Crystal
	:members:
	:special-members:
	:show-inheritance:
	
.. autoclass:: pynlo.media.crystals.PPLN.PPLN
	:members:
	:special-members:
	:show-inheritance:

.....More undocumented crystals here....




pynlo.util.ode_solve
--------------------
These classes are an adaptation of the very nice *Numerical Recipes* ODE solvers into Python. The solver is divided into two parts: specific step iterators (eg Dopri853) and the framework for stepping through the ODE (steppers)


Dormand-Prince 853 Stepper
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pynlo.util.ode_solve.dopr853.StepperDopr853
    :members:
	:special-members:
    :undoc-members:
    :show-inheritance:

Steppers and helpers
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pynlo.util.ode_solve.steppers.Output
	:members:
	:special-members:
	:show-inheritance:
	
.. autoclass:: pynlo.util.ode_solve.steppers.StepperBase
	:members:
	:special-members:
	:show-inheritance:
	
.. autoclass:: pynlo.util.ode_solve.steppers.ODEint
	:members:
	:special-members:
	:show-inheritance:



pynlo.devices
-------------

.. autoclass:: pynlo.devices.grating_compressor.TreacyCompressor
    :members:
    :show-inheritance:
	
	
