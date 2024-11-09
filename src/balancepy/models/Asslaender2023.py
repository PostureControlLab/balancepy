import numpy as np
from scipy.optimize import basinhopping
import balancepy.biomechanics as bm
from helper import dict2par

def settings(BM=72,BH=1.84):
    # Scale model parameters to anthropometric data
    WT = bm.WinterTable(BM,BH)
    mgh = WT.mgh / 180*np.pi
    J = WT.J / 180*np.pi

    opts = {
        'name': 'Asslaender2023',
        'description': 'IC Model fit with maximum likelihood objective function. From Publication: doi...',
        'modelEq': 'dynamics_Asslaender2023(f,p)',
        'objectiveFunc': 'objective_Asslaender2023(theta, keys_fit, model_as_func, parameter_fix, TF_exp)',
        'parameter_default': {
            # default/start, fit bool, lower bound, upper bound
            'mgh':  mgh,
            'J':    J,
            'Kp':   1.15 * mgh,
            'Kd':   0.3 * mgh,
            'W':    0.2,
            'dt':   0.19,
            'Glp':  0.1,
            'Flp':  20,
            'b':   1
        },
        'parameter_fit': {
            # fit bool, lower bound, upper bound
            'mgh':  [False, 10,  20],
            'J':    [False, 0,   0],
            'Kp':   [True,  mgh,   2*mgh],
            'Kd':   [True,  0, 1*mgh],
            'W':    [True,  0.01,   1],
            'dt':   [True,  0.05,   0.3],
            'Glp':  [True,  0,   0.3],
            'Flp':  [False, 3,   30],
            'b':    [True, 0.0001,   10]
        }
        }

    return opts

def dynamics(f = np.linspace(0.05, 2, 100) , p =settings()['parameter_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = (p['Glp'] * p['Flp']) / (p['Flp'] * s +1)
    
    tf = (p['W'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def objective(theta, keys_fit, model_as_func, parameter_fix, TF_exp):  
    # pair the keys with the theta values for better readability of model dynamics
    p = dict(zip(keys_fit, theta))
    p.update(parameter_fix)

    tf = model_as_func(p)
    
    err = np.sum(np.log(2 * p['b'] * np.abs(tf))) + np.sum(np.abs(tf - TF_exp) / (p['b'] * np.abs(tf)))
    return err

def fit(FD, opts = settings()):
    f = FD['f']
    model_as_func = lambda p: dynamics(f,p)

    TF_exp = FD['FRF']
    theta_start, bounds, keys_fit, parameter_fix = par2dict(opts)
    obj = lambda theta: objective(theta, keys_fit, model_as_func, parameter_fix, TF_exp)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(obj, theta_start, minimizer_kwargs=minimizer_kwargs)

    parameter_out = dict(zip(keys_fit, res.x))
    parameter_out.update(parameter_fix)
    tf_sim, f = model_as_func(parameter_out)

    return parameter_out, tf_sim, res

