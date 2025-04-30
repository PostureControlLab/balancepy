
class Parameter:
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
            return {name: p.value for name, p in self._params.items() if not p.fixed}
        return {name: p.value for name, p in self._params.items()}