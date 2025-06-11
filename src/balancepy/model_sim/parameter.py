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
        self.multicond_name = name # For multi-condition models, this can be used to differentiate parameters

    def __repr__(self):
        return f"Parameter(name={self.name}, value={self.value}, bounds={self.bounds}, fixed={self.fixed}, unit={self.unit}, description={self.description})"

class ParameterSet:
    """
    Manages a collection of Parameter objects for a model.

    This class enables organized handling of model parameters, including adding, accessing, updating,
    and retrieving parameter information such as values, bounds, units, and descriptions. It supports
    operations for both fixed and free parameters, making it suitable for parameter management in
    modeling and simulation tasks.
    """
    def __init__(self):
        self._params = {}
        self._multicond_lookup = {}

    def add(self, param: Parameter):
        self._params[param.name] = param
        self._multicond_lookup[param.multicond_name] = param

    def get_by_multicond_name(self, multicond_name):
        return self._multicond_lookup[multicond_name]

    def set_by_multicond_name(self, multicond_name, value):
        self._multicond_lookup[multicond_name].value = value

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
        # Create a summary of all parameters, each on a new line, with rounded values
        param_summaries = [
            f"{name}: value={round(param.value, 2) if isinstance(param.value, (float, int)) else param.value}, bounds={param.bounds}, fixed={param.fixed}"
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
    
    def update_multicond_name(self, old_name, new_name):
        # Remove old mapping
        param = self._multicond_lookup.pop(old_name)
        # Update the parameter's multicond_name attribute
        param.multicond_name = new_name
        # Add new mapping
        self._multicond_lookup[new_name] = param