import numpy as np
from scipy.optimize import Bounds, basinhopping
import balancepy.biomechanics as bm
from balancepy.models.helper import dict2par


def settings(body_mass=72,body_height=1.84):
    # Scale model parameters to anthropometric data
    WT = bm.WinterTable(body_mass,body_height)
    mgh = WT.mgh / 180*np.pi
    J = WT.J / 180*np.pi

    opts = {
        'name': 'ICmodel_Peterka2018',
        'description': 'Model as described in Peterka et al. 2018',
        'modelEq': 'dynamics(f,p)',
        'objectiveFunc': 'objective(theta, keys_fit, model_as_func, parameter_fix, TF_exp)',
        'parameter_default': {
            # default/start, fit bool, lower bound, upper bound
            'mgh':  mgh,
            'J':    J,
            'Kp':   1.15 * mgh,
            'Kd':   0.3 * mgh,
            'W':    0.2,
            'dt':   0.19,
            'Glp':  0.1
        },
        'parameter_fit': {
            # fit bool, lower bound, upper bound
            'mgh':  [False, 10,  20],
            'J':    [False, 0,   0],
            'Kp':   [True,  mgh,   2*mgh],
            'Kd':   [True,  0, 1*mgh],
            'W':    [True,  0.01,   1],
            'dt':   [True,  0.05,   0.3],
            'Glp':  [True,  0,   0.3]
        }
        }

    return opts

def dynamics(f = np.linspace(0.05, 2, 100) , p =settings()['parameter_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = p['Glp'] / s
    
    tf = (p['W'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf

def objective(theta, keys_fit, model_as_func, parameter_fix, TF_exp):  
    p = dict(zip(keys_fit, theta))
    p.update(parameter_fix)

    tf = model_as_func(p)
    
    err = np.sum( np.abs(tf - TF_exp) / (np.abs(tf)) )

    return err

def fit(FD, opts = settings()):
    f = FD['f']
    model_as_func = lambda p: dynamics(f,p)

    TF_exp = FD['FRF']
    theta_start, bounds, keys_fit, parameter_fix = dict2par(opts)
    obj = lambda theta: objective(theta, keys_fit, model_as_func, parameter_fix, TF_exp)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(obj, theta_start, minimizer_kwargs=minimizer_kwargs)

    parameter_out = dict(zip(keys_fit, res.x))
    parameter_out.update(parameter_fix)
    tf_sim = model_as_func(parameter_out)

    return parameter_out, tf_sim, res


