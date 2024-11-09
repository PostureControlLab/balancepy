import numpy as np
from scipy.optimize import Bounds, basinhopping
import balancepy.biomechanics as bm
from helper import dict2par

def settings(body_mass=72,body_height=1.84):
    # Scale model parameters to anthropometric data
    WT = bm.WinterTable(body_mass,body_height)
    mgh = WT.mgh / 180*np.pi
    J = WT.J / 180*np.pi

    opts = {
        'name': 'ICdual',
        'description': 'Independent Channel model with dual surface and visual tilt stimuli',
        'modelEq': 'dynamics(f,p)',
        'objectiveFunc': 'objective(theta, keys_fit, model_as_func, parameter_fix, TF_exp)',
        'parameter_default': {
            # default/start, fit bool, lower bound, upper bound
            'mgh':  mgh,
            'J':    J,
            'Kp':   1.15 * mgh,
            'Kd':   0.3 * mgh,
            'Wv':    0.2,
            'Wp':    0.2,
            'dt':   0.19,
            'Glp':  0.1
        },
        'parameter_fit': {
            # fit bool, lower bound, upper bound
            'mgh':  [False, 10,  20],
            'J':    [False, 0,   0],
            'Kp':   [True,  mgh,   2*mgh],
            'Kd':   [True,  0, 1*mgh],
            'Wv':    [True,  0.01,   1],
            'Wp':    [True,  0.01,   1],
            'dt':   [True,  0.05,   0.3],
            'Glp':  [True,  0,   0.3]
        }
        }

    return opts

def dynamics_prop(f = np.linspace(0.05, 2, 100) , p =settings()['parameter_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = (p['Glp'] * p['Flp']) / (p['Flp'] * s +1)
    
    tf = (p['Wp'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def dynamics_vis(f = np.linspace(0.05, 2, 100) , p =settings()['parameter_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = (p['Glp'] * p['Flp']) / (p['Flp'] * s +1)
    
    tf = (p['Wv'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def objective(theta, keys_fit, fdyn_prop, fdyn_vis, parameter_fix, TF_prop, TF_vis):  
    p = dict(zip(keys_fit, theta))
    p.update(parameter_fix)

    tf_prop = fdyn_prop(p)
    err_prop = np.sum(np.log(2 * p['b'] * np.abs(tf_prop))) + np.sum(np.abs(tf_prop - TF_prop) / (p['b'] * np.abs(tf_prop)))
    
    tf_vis = fdyn_vis(p)
    err_vis = np.sum(np.log(2 * p['b'] * np.abs(tf_vis))) + np.sum(np.abs(tf_vis - TF_vis) / (p['b'] * np.abs(tf_vis)))
    
    err = err_prop + err_vis

    return err

def fit(FD_prop, FD_vis, opts = settings()):

    fdyn_prop = lambda p: dynamics_prop(FD_prop['f'],p)
    fdyn_vis  = lambda p: dynamics_vis(FD_vis['f'],p)

    theta, bounds, keys_fit, parameter_fix = dict2par(opts)

    obj = lambda theta: objective(theta, keys_fit, fdyn_prop, fdyn_vis, parameter_fix, FD_prop['FRF'], FD_vis['FRF'])

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(obj, theta, minimizer_kwargs=minimizer_kwargs)

    parameter_out = dict(zip(keys_fit, res.x))
    parameter_out.update(parameter_fix)

    tf_sim_prop, f_prop = fdyn_prop(parameter_out)
    tf_sim_vis, f_vis = fdyn_vis(parameter_out)

    return parameter_out, tf_sim_prop, f_prop, tf_sim_vis, f_vis, res