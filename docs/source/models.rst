Model simulations
=================

BaseModel
---------

.. autoclass:: balancepy.model_sim.base_model.BaseModel
   :undoc-members:
   :show-inheritance:

Attributes
~~~~~~~~~~

+----------------+-----------------------------------------------+
| Name           | Description                                   |
+================+===============================================+
| ModelName      | Name of the model.                            |
+----------------+-----------------------------------------------+
| mass_kg        | Subject mass in kilograms.                    |
+----------------+-----------------------------------------------+
| height_m       | Subject height in meters.                     |
+----------------+-----------------------------------------------+
| data_exp       | Experimental data object (`sr_data` or None). |
+----------------+-----------------------------------------------+
| data_sim       | Simulated data object (`sr_data` or None).    |
+----------------+-----------------------------------------------+
| params         | Model parameters (`ParameterSet`).            |
+----------------+-----------------------------------------------+
| fit_output     | Output of the fitting procedure.              |
+----------------+-----------------------------------------------+
| dynamics       | Defines the system dynamics (callable).       |
+----------------+-----------------------------------------------+

Methods
~~~~~~~

+------------------------+-------------------------------------------------------------+
| Name                   | Description                                                 |
+========================+=============================================================+
| frf(freq=None)         | Computes the frequency response function (FRF).             |
+------------------------+-------------------------------------------------------------+
| simulate_timedomain    | Simulates the time domain response of the system.           |
| (stimulus,             |                                                             |
| samplingrate_Hz)       |                                                             |
+------------------------+-------------------------------------------------------------+
| objective(params_free) | Objective function for optimization.                        |
+------------------------+-------------------------------------------------------------+
| fit(data_exp=None)     | Fits the model parameters to experimental data.             |
+------------------------+-------------------------------------------------------------+
| plot()                 | Plots the experimental and simulated data, if available.    |
+------------------------+-------------------------------------------------------------+


Specific Models
----------------

.. autoclass:: balancepy.model_sim.p18.P18
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: balancepy.model_sim.a23.A23
   :members:
   :undoc-members:
   :show-inheritance:

Parameter Classes
-----------------

.. autoclass:: balancepy.model_sim.parameter.Parameter
   :undoc-members:
   :show-inheritance:

Attributes
~~~~~~~~~~

+-------------------+---------------------------------------------------------------+
| Name              | Description                                                   |
+===================+===============================================================+
| name              | The name of the parameter.                                    |
+-------------------+---------------------------------------------------------------+
| value             | The current value of the parameter.                           |
+-------------------+---------------------------------------------------------------+
| default           | The default value of the parameter.                           |
+-------------------+---------------------------------------------------------------+
| bounds            | The lower and upper bounds of the parameter (tuple).          |
+-------------------+---------------------------------------------------------------+
| fixed             | Whether the parameter is fixed (not optimized).               |
+-------------------+---------------------------------------------------------------+
| fit_result        | Result from fitting, if applicable.                           |
+-------------------+---------------------------------------------------------------+
| confidencebounds  | Confidence bounds for the parameter, if applicable.           |
+-------------------+---------------------------------------------------------------+
| unit              | The unit of the parameter, if applicable.                     |
+-------------------+---------------------------------------------------------------+
| description       | A description of the parameter, if applicable.                |
+-------------------+---------------------------------------------------------------+

.. autoclass:: balancepy.model_sim.parameter.ParameterSet
   :undoc-members:
   :show-inheritance:

Methods
~~~~~~~

+-----------------------------+---------------------------------------------------------------+
| Name                        | Description                                                   |
+=============================+===============================================================+
| add(param)                  | Add a `Parameter` object to the set.                          |
+-----------------------------+---------------------------------------------------------------+
| items()                     | Return (name, Parameter) pairs.                               |
+-----------------------------+---------------------------------------------------------------+
| names()                     | List of parameter names.                                      |
+-----------------------------+---------------------------------------------------------------+
| defaults()                  | Dictionary of default values for all parameters.              |
+-----------------------------+---------------------------------------------------------------+
| units()                     | Dictionary of parameter units.                                |
+-----------------------------+---------------------------------------------------------------+
| descriptions()              | Dictionary of parameter descriptions.                         |
+-----------------------------+---------------------------------------------------------------+
| values(only_free=True)      | List of parameter values, optionally only for free parameters.|
+-----------------------------+---------------------------------------------------------------+
| bounds()                    | List of bounds for free (not fixed) parameters.               |
+-----------------------------+---------------------------------------------------------------+
| set_values(values,          | Set parameter values from a list, optionally only for free    |
| only_free=True)             |parameters.                                                    |
+-----------------------------+---------------------------------------------------------------+
| set_defaults()              | Reset all parameter values to their defaults.                 |
+-----------------------------+---------------------------------------------------------------+
| to_value_dict(              | Dictionary of parameter values, optionally only for free      |
| only_free=False)            |parameters.                                                    |
+-----------------------------+---------------------------------------------------------------+

