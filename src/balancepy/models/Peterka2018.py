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

def dynamics(f = np.linspace(0.05, 2, 100) , p =getOpts_ICmodel_Peterka2018()['parameter_default']):
    
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
    theta_start, bounds, keys_fit, parameter_fix = convert_parameter_for_fit(opts)
    obj = lambda theta: objective(theta, keys_fit, model_as_func, parameter_fix, TF_exp)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(obj, theta_start, minimizer_kwargs=minimizer_kwargs)

    parameter_out = dict(zip(keys_fit, res.x))
    parameter_out.update(parameter_fix)
    tf_sim = model_as_func(parameter_out)

    return parameter_out, tf_sim, res


def getOpts_ICmodel_maxLikelihood(BM=72,BH=1.84):
    # Scale model parameters to anthropometric data
    WT = bm.WinterTable(BM,BH)
    mgh = WT.mgh / 180*np.pi
    J = WT.J / 180*np.pi

    opts = {
        'name': 'ICmodel_maxLikelihood',
        'description': 'IC Model fit with maximum likelihood objective',
        'modelEq': 'run_ICmodel_maxLikelihood(f,p)',
        'objectiveFunc': 'objective_ICmodel_maxLikelihood(theta, keys_fit, model_as_func, parameter_fix, TF_exp)',
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

def run_ICmodel_maxLikelihood(f = np.linspace(0.05, 2, 100) , p =getOpts_ICmodel_Peterka2018()['parameter_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = (p['Glp'] * p['Flp']) / (p['Flp'] * s +1)
    
    tf = (p['W'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def objective_ICmodel_maxLikelihood(theta, keys_fit, model_as_func, parameter_fix, TF_exp):  
    p = dict(zip(keys_fit, theta))
    p.update(parameter_fix)

    tf = model_as_func(p)
    
    err = np.sum(np.log(2 * p['b'] * np.abs(tf))) + np.sum(np.abs(tf - TF_exp) / (p['b'] * np.abs(tf)))
    return err

def fit_ICmodel_maxLikelihood(FD, opts = getOpts_ICmodel_Peterka2018()):
    f = FD['f']
    model_as_func = lambda p: run_ICmodel_maxLikelihood(f,p)

    TF_exp = FD['FRF']
    theta_start, bounds, keys_fit, parameter_fix = convert_parameter_for_fit(opts)
    obj = lambda theta: objective_ICmodel_maxLikelihood(theta, keys_fit, model_as_func, parameter_fix, TF_exp)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(obj, theta_start, minimizer_kwargs=minimizer_kwargs)

    parameter_out = dict(zip(keys_fit, res.x))
    parameter_out.update(parameter_fix)
    tf_sim, f = model_as_func(parameter_out)

    return parameter_out, tf_sim, res

def run_dual_IndependentChannelModel(f = np.linspace(0.05, 2, 100) , p =getOpts_ICmodel_Peterka2018()['parameter_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = (p['Glp'] * p['Flp']) / (p['Flp'] * s +1)
    
    tf = (p['W'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def objective_dual_IndependentChannelModel(theta, keys_fit, model_as_func, parameter_fix, TF_exp):  
    p = dict(zip(keys_fit, theta))
    p.update(parameter_fix)

    tf = model_as_func(p)
    
    err = np.sum(np.log(2 * p['b'] * np.abs(tf))) + np.sum(np.abs(tf - TF_exp) / (p['b'] * np.abs(tf)))
    return err

def fit_dual_IndependentChannelModel(FD1,FD2, opts = getOpts_ICmodel_Peterka2018()):
    f1 = FD1['f']
    f2 = FD2['f']
    model_as_func = lambda p: run_dual_IndependentChannelModel(f,p)

    TF_exp = FD1['FRF']
    theta_start, bounds, keys_fit, parameter_fix = convert_parameter_for_fit(opts)
    obj = lambda theta: objective_ICmodel_maxLikelihood(theta, keys_fit, model_as_func, parameter_fix, TF_exp)

    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    res = basinhopping(obj, theta_start, minimizer_kwargs=minimizer_kwargs)

    parameter_out = dict(zip(keys_fit, res.x))
    parameter_out.update(parameter_fix)
    tf_sim, f = model_as_func(parameter_out)

    return parameter_out, tf_sim, res



def convert_parameter_for_fit(fit_opts):
    parameter_fix = {}
    theta = []
    lb = []
    ub = []
    keys_fit = []

    for key, val in fit_opts['parameter_fit'].items():
        if val[0]:
            theta.append(fit_opts['parameter_default'][key])
            lb.append(val[1])
            ub.append(val[2])
            keys_fit.append(key)
        else:
            parameter_fix[key] = fit_opts['parameter_default'][key]

    bounds = Bounds(lb, ub)

    return theta, bounds, keys_fit, parameter_fix


def parfit(fit_opts):
    theta, bounds, keys_fit, parameter_fix = convert_parameter_for_fit(fit_opts)

    model_as_func = eval('lambda p: ' + expression)

    #res = basinhopping(objectiveFunc, theta, args=(TFexp,fit_opts), bounds = bounds, niter=100)
    res = minimize(objectiveFunc, theta, args=(fit_opts,parameter_fix), bounds=bounds)
    theta_out = res.x

    parameter_fit = parameter_fix
    parameter_fit.update(dict(zip(keys_fit, theta_out)))

    return parameter_fit, res