from scipy.optimize import Bounds

def dict2par(settings):
    parameter_fix = {}
    theta = []
    lb = []
    ub = []
    keys_fit = []

    for key, val in settings['parameter_fit'].items():
        if val[0]:
            theta.append(settings['parameter_default'][key])
            lb.append(val[1])
            ub.append(val[2])
            keys_fit.append(key)
        else:
            parameter_fix[key] = settings['parameter_default'][key]

    bounds = Bounds(lb, ub)

    return theta, bounds, keys_fit, parameter_fix