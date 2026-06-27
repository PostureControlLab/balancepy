import numpy as np
from scipy.optimize import basinhopping
from balancepy.model_sim.base_model import BaseModel
from balancepy.model_sim.parameter import ParameterSet, Parameter
import plotly.graph_objects as go

class MultiModelParameterSet(ParameterSet):
    """
    A specialized ParameterSet for managing parameters across multiple models.
    
    This class extends ParameterSet to handle parameters that are shared across multiple models,
    allowing for organized management and retrieval of parameters in a multi-model context.
    """
    def __init__(self):
        super().__init__()

    # Remove attributes from ParameterSet
    def set_values(self, *args, **kwargs):
        raise NotImplementedError("set_values is disabled for MultiModelParameterSet.")
    def set_defaults(self, *args, **kwargs):
        raise NotImplementedError("set_defaults is disabled for MultiModelParameterSet.")
    def to_value_dict(self, *args, **kwargs):
        raise NotImplementedError("to_value_dict is disabled for MultiModelParameterSet.")
    def get_by_multimodel_name(self, *args, **kwargs):
        raise NotImplementedError("get_by_multimodel_name is disabled for MultiModelParameterSet.")
    def set_by_multimodel_name(self, *args, **kwargs):
        raise NotImplementedError("set_by_multimodel_name is disabled for MultiModelParameterSet.")
    def update_multimodel_name(self, *args, **kwargs):
        raise NotImplementedError("update_multimodel_name is disabled for MultiModelParameterSet.")

class MultiModel(BaseModel):
    def __init__(self, model_list):
        self.model_list = model_list
        self.fit_output = None
        self.params = MultiModelParameterSet()

        # Collect all unique multimodel parameters across models
        for model in self.model_list:
            for param in model.params:
                mm_name = param.multimodel_name
                # Add a new parameter to the MultiModelParameterSet if mm_name is not already present
                if mm_name not in self.params.names():
                    self.params.add(Parameter(mm_name, param.value, bounds=param.bounds, fixed=param.fixed, unit=param.unit, description=param.description))
                else:
                    # Warn if bounds/fixed differ
                    if param.value != self.params[mm_name].value:
                        print(f"Warning: Initial value for parameter '{mm_name}' does not match in all models.")
                    if param.bounds[0] != self.params[mm_name].bounds[0]:
                        print(f"Warning: Lower bound for parameter '{mm_name}' does not match in all models.")
                    if param.bounds[1] != self.params[mm_name].bounds[1]:
                        print(f"Warning: Upper bound for parameter '{mm_name}' does not match in all models.")


    def set_mm_parameter(self, param_name, value):
        """
        Set the value of a single parameter by its multimodel name.
        
        Parameters
        ----------
        param_name : str
            The multimodel name of the parameter.
        value : float
            The value to set for the parameter.
        """
        if param_name in self.params.names():
            self.params[param_name].value = value
            for model in self.model_list:
                model.params.set_by_multimodel_name(param_name, value)
        else:
            raise KeyError(f"Parameter '{param_name}' not found in MultiModelParameterSet.")

    def set_values(self, values, only_free=True):
        """
        Set values of all parameters in the MultiModelParameterSet.

        Parameters
        ----------
        values : list or np.ndarray
            The values to set for the parameters.
        only_free : bool
            If True, only set values for parameters that are not fixed.
        """
        i = 0
        for p in self.params:
            if not only_free or not p.fixed:
                # Update MultiModel parameter value
                p.value = values[i]

                # Update each model's parameter by its multimodel name
                for model in self.model_list:
                    if p.name in model.params._multimodel_lookup:
                        if not model.params.get_by_multimodel_name(p.name).fixed:
                            model.params.set_by_multimodel_name(p.name, values[i])
                i += 1

    def set_defaults(self):
        """
        Reset all parameters in the MultiModelParameterSet to their default values.
        """
        for p in self.params:
            p.value = p.default

            # Reset each model's parameter by its multimodel name
            for model in self.model_list:
                if p.name in model.params._multimodel_lookup:
                    model.params.set_by_multimodel_name(p.name, p.default)
                    
    def objective(self, params_free = None):
        # Set parameters if changed e.g. during fitting
        if params_free is not None:
            self.set_values(params_free, only_free=True)

        # Calculate the objective function as the sum of errors across all models
        # This assumes each model's objective returns a scalar error value
        err = 0
        for model in self.model_list:
            err += model.objective()
        return err

    def fit(self):
        # Set initial guess for free paramss
        theta_free_init = self.params.values(only_free=True)

        bounds = self.params.bounds()
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        self.fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)

        params_fit = self.fit_output.x

        self.set_values(params_fit, only_free=True)

        # Update the simulated data object with the system behavior after fitting
        for model in self.model_list:
            model._update_data_sim()

        return self.fit_output.x, self.fit_output

    def plot_param_table(self):
        # Prepare header: first column is 'Model', then all multimodel_names
        header = ['MultiModel'] + self.params.names()
    
        # Prepare fixed_row for header: None for first column, then fixed status for each parameter
        row = ['Values'] + [np.round(self.params[name].value,3) for name in self.params.names()]
        fixed_row = [None] + [self.params[name].fixed for name in self.params.names()]
    
        # Prepare rows and fixed_rows
        rows = [row]
        fixed_rows = [fixed_row]
    
        for model in self.model_list:
            row = [model.ModelName]
            fixed_row = [None]
            # Build a lookup: mm_name -> param.name for this model
            param_lookup = {param.multimodel_name: param.name for param in model.params}
            for mm_name in self.params.names():
                row.append(param_lookup.get(mm_name, ''))
                param_obj = model.params.get_by_multimodel_name(mm_name)
                fixed_row.append(param_obj.fixed if param_obj is not None else None)
            rows.append(row)
            fixed_rows.append(fixed_row)
    
        # Convert to columns for plotly
        columns = list(map(list, zip(*rows)))
    
        # Determine cell colors: white for model name, red for fixed, green for free
        cell_colors = [
            ['white'] + ['#ffcccc' if is_fixed else '#ccffcc' for is_fixed in fixed_row[1:]]
            for fixed_row in fixed_rows
        ]
        # Transpose to columns for plotly
        cell_colors = list(map(list, zip(*cell_colors)))
    
        fig = go.Figure(data=[go.Table(
            header=dict(values=header, fill_color='paleturquoise', align='left'),
            cells=dict(values=columns, fill_color=cell_colors, align='left'),
            columnwidth=[120] + [80]*len(self.params.names())  # Set first column wider, others narrower
        )])
        fig.update_layout(title='MultiModel Parameter Mapping')
        fig.show()