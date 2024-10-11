import numpy as np
from scipy.optimize import Bounds, minimize



def parfit(fit_opts):
    theta, bounds, keys_fit, par_fix = convert_par_for_fit(fit_opts)

    model_as_func = eval('lambda p: ' + expression)

    #res = basinhopping(objectiveFunc, theta, args=(TFexp,fit_opts), bounds = bounds, niter=100)
    res = minimize(objectiveFunc, theta, args=(fit_opts,par_fix), bounds=bounds)
    theta_out = res.x

    par_fit = par_fix
    par_fit.update(dict(zip(keys_fit, theta_out)))

    return par_fit, res



