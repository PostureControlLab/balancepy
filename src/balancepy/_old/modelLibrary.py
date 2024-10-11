import numpy as np
import biomechanics as bm
from scipy.optimize import Bounds, basinhopping

def getOpts_ICmodel_Peterka2018(BM=72,BH=1.84):
    # Scale model parameters to anthropometric data
    WT = bm.winter_table(BM,BH)
    mgh = WT['mgh'] / 180*np.pi
    J = WT['J'] / 180*np.pi

    opts = {
        'name': 'ICmodel_Peterka2018',
        'description': 'Model as described in Peterka et al. 2018',
        'modelEq': 'run_ICmodel_Peterka2018(f,p)',
        'objectiveFunc': 'objective_ICmodel_Peterka2018(theta, keys_fit, model_as_func, par_fix, TF_exp)',
        'par_default': {
            # default/start, fit bool, lower bound, upper bound
            'mgh':  mgh,
            'J':    J,
            'Kp':   1.15 * mgh,
            'Kd':   0.3 * mgh,
            'W':    0.2,
            'dt':   0.19,
            'Glp':  0.1
        },
        'par_fit': {
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

def run_ICmodel_Peterka2018(f = np.linspace(0.05, 2, 100) , p =getOpts_ICmodel_Peterka2018()['par_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = p['Glp'] / s
    
    tf = (p['W'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def objective_ICmodel_Peterka2018(theta, keys_fit, model_as_func, par_fix, TF_exp):  
    p = dict(zip(keys_fit, theta))
    p.update(par_fix)

    tf = model_as_func(p)
    
    err = np.sum( np.abs(tf - TF_exp) / (np.abs(tf)) )

    return err

def fit_ICmodel_Peterka2018(FD, opts = getOpts_ICmodel_Peterka2018()):
    f = FD['f']
    model_as_func = lambda p: run_ICmodel_Peterka2018(f,p)

    TF_exp = FD['FRF']
    theta_start, bounds, keys_fit, par_fix = convert_par_for_fit(opts)
    obj = lambda theta: objective_ICmodel_Peterka2018(theta, keys_fit, model_as_func, par_fix, TF_exp)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(obj, theta_start, minimizer_kwargs=minimizer_kwargs)

    par_out = dict(zip(keys_fit, res.x))
    par_out.update(par_fix)
    tf_sim, f = model_as_func(par_out)

    return par_out, tf_sim, res


def getOpts_ICmodel_maxLikelihood(BM=72,BH=1.84):
    # Scale model parameters to anthropometric data
    WT = bm.winter_table(BM,BH)
    mgh = WT['mgh'] / 180*np.pi
    J = WT['J'] / 180*np.pi

    opts = {
        'name': 'ICmodel_Peterka2018',
        'description': 'Model as described in Peterka et al. 2018',
        'modelEq': 'run_ICmodel_Peterka2018(f,p)',
        'objectiveFunc': 'objective_ICmodel_Peterka2018(theta, keys_fit, model_as_func, par_fix, TF_exp)',
        'par_default': {
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
        'par_fit': {
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

def run_ICmodel_maxLikelihood(f = np.linspace(0.05, 2, 100) , p =getOpts_ICmodel_Peterka2018()['par_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = (p['Glp'] * p['Flp']) / (p['Flp'] * s +1)
    
    tf = (p['W'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def objective_ICmodel_maxLikelihood(theta, keys_fit, model_as_func, par_fix, TF_exp):  
    p = dict(zip(keys_fit, theta))
    p.update(par_fix)

    tf = model_as_func(p)
    
    err = np.sum(np.log(2 * p['b'] * np.abs(tf))) + np.sum(np.abs(tf - TF_exp) / (p['b'] * np.abs(tf)))
    return err

def fit_ICmodel_maxLikelihood(FD, opts = getOpts_ICmodel_Peterka2018()):
    f = FD['f']
    model_as_func = lambda p: run_ICmodel_maxLikelihood(f,p)

    TF_exp = FD['FRF']
    theta_start, bounds, keys_fit, par_fix = convert_par_for_fit(opts)
    obj = lambda theta: objective_ICmodel_maxLikelihood(theta, keys_fit, model_as_func, par_fix, TF_exp)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(obj, theta_start, minimizer_kwargs=minimizer_kwargs)

    par_out = dict(zip(keys_fit, res.x))
    par_out.update(par_fix)
    tf_sim, f = model_as_func(par_out)

    return par_out, tf_sim, res



def convert_par_for_fit(fit_opts):
    par_fix = {}
    theta = []
    lb = []
    ub = []
    keys_fit = []

    for key, val in fit_opts['par_fit'].items():
        if val[0]:
            theta.append(fit_opts['par_default'][key])
            lb.append(val[1])
            ub.append(val[2])
            keys_fit.append(key)
        else:
            par_fix[key] = fit_opts['par_default'][key]

    bounds = Bounds(lb, ub)

    return theta, bounds, keys_fit, par_fix

