
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
        return f"Parameter(name={self.name}, default={self.default}, bounds={self.bounds})"

class ParameterSet:
    def __init__(self):
        self._params = {}

    def add(self, param: Parameter):
        self._params[param.name] = param

    def __getitem__(self, name):
        return self._params[name]

    def __iter__(self):
        return iter(self._params.values())

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

    def to_value_dict(self, only_free=False):
        if only_free:
            return {name: p.value for name, p in self._params.items() if not p.fixed}
        return {name: p.value for name, p in self._params.items()}