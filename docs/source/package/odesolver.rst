Numerical Recipes-based ODESolve
--------------------------------

These classes are an adaptation of the very nice *Numerical Recipes* ODE solvers into Python. The solver is divided into two parts: specific step iterators (eg Dopri853) and the framework for stepping through the ODE (steppers)

Steppers and helpers
--------------------

.. autoclass:: pynlo.util.ode_solve.steppers.Output
	:members:
	:show-inheritance:
	
.. autoclass:: pynlo.util.ode_solve.steppers.StepperBase
	:members:
	:show-inheritance:
	
.. autoclass:: pynlo.util.ode_solve.steppers.ODEint
	:members:
	:show-inheritance:

Dormand-Prince 853 Stepper
--------------------------
.. autoclass:: pynlo.util.ode_solve.dopr853.StepperDopr853
	:members:
	:show-inheritance:

