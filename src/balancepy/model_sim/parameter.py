# class Parameter and ParameterSet for managing model parameters in balancepy
class Parameter:
    """
    Represents a single parameter in a model.

    Parameters
    ----------
    name : str
        The name of the parameter.
    default : float
        The default value of the parameter.
    bounds : tuple, optional
        A tuple representing the lower and upper bounds of the parameter (default is (None, None)).
    fixed : bool, optional
        Whether the parameter is fixed (not optimized). Default is False.
    unit : str, optional
        The unit of the parameter, if applicable.
    description : str, optional
        A description of the parameter, if applicable.

    Attributes
    ----------
    name : str
        The name of the parameter.
    value : float
        The current value of the parameter.
    default : float
        The default value of the parameter.
    bounds : tuple
        The lower and upper bounds of the parameter.
    fixed : bool
        Whether the parameter is fixed (not optimized).
    fit_result : object or None
        Result from fitting, if applicable.
    confidencebounds : object or None
        Confidence bounds for the parameter, if applicable.
    unit : str or None
        The unit of the parameter, if applicable.
    description : str or None
        A description of the parameter, if applicable.
    """
    def __init__(self, name, default, bounds=(None, None), fixed=False, unit=None, description=None):
        self.name = name
        self.value = default
        self.default = default
        self.bounds = bounds
        self.fixed = fixed
        self.fit_result = None
        self.confidencebounds = None
        self.unit = unit
        self.description = description

    def __repr__(self):
        return f"Parameter(name={self.name}, value={self.value}, bounds={self.bounds}, fixed={self.fixed}, unit={self.unit}, description={self.description})"

class ParameterSet:
    """
    Manages a collection of Parameter objects for a model.

    This class enables organized handling of model parameters, including adding, accessing, updating,
    and retrieving parameter information such as values, bounds, units, and descriptions. It supports
    operations for both fixed and free parameters, making it suitable for parameter management in
    modeling and simulation tasks.

    Attributes
    ----------
    _params : dict
        Dictionary mapping parameter names to Parameter instances.

    Methods
    -------
    add(param)
        Add a Parameter object to the set.
    items()
        Return (name, Parameter) pairs.
    names()
        List of parameter names.
    defaults()
        Dictionary of default values for all parameters.
    units()
        Dictionary of parameter units.
    descriptions()
        Dictionary of parameter descriptions.
    values(only_free=True)
        List of parameter values, optionally only for free (not fixed) parameters.
    bounds()
        List of bounds for free (not fixed) parameters.
    set_values(values, only_free=True)
        Set parameter values from a list, optionally only for free parameters.
    set_defaults()
        Reset all parameter values to their defaults.
    to_value_dict(only_free=False)
        Dictionary of parameter values, optionally only for free parameters.
    """
    def __init__(self):
        self._params = {}

    def add(self, param: Parameter):
        self._params[param.name] = param

    def __getitem__(self, name):
        return self._params[name]
    
    def __setitem__(self, name, value):
        if name in self._params:
            self._params[name].value = value  # Update the value of the existing parameter
        else:
            raise KeyError(f"Parameter '{name}' does not exist in the ParameterSet.")

    def __iter__(self):
        return iter(self._params.values())

    def __repr__(self):
        # Create a summary of all parameters, each on a new line
        param_summaries = [
            f"{name}: value={param.value}, bounds={param.bounds}, fixed={param.fixed}"
            for name, param in self._params.items()
        ]
        return "ParameterSet( \n    " + ",\n    ".join(param_summaries) + "\n    )"

    def items(self):
        return self._params.items()

    def names(self):
        return list(self._params.keys())

    def defaults(self):
        return {name: param.default for name, param in self._params.items()}

    def units(self):
        return {name: param.unit for name, param in self._params.items()}

    def descriptions(self):
        return {name: param.description for name, param in self._params.items()}
    
    def values(self, only_free=True):
        return [
            p.value for p in self._params.values()
            if not only_free or not p.fixed
        ]

    def bounds(self):
        return [
            p.bounds for p in self._params.values()
            if not p.fixed
        ]

    def set_values(self, values, only_free=True):
        i = 0
        for p in self._params.values():
            if not only_free or not p.fixed:
                p.value = values[i]
                i += 1

    def set_defaults(self):
        for p in self._params.values():
            p.value = p.default

    def to_value_dict(self, only_free=False):
        if only_free:
            return {name: float(p.value) for name, p in self._params.items() if not p.fixed}
        return {name: float(p.value) for name, p in self._params.items()}