import numpy as np
from scipy.optimize import Bounds, basinhopping
import balancepy.biomechanics as bm
from balancepy.models.helper import dict2par
import balancepy.frequency as fd

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
            'Glp':  0.1,
            'b':   1
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
            'Glp':  [True,  0,   0.3],
            'b':    [True, 0.0001,   10]
            }
        }

    return opts

def dynamics_prop(f = np.linspace(0.05, 2, 100) , p =settings()['parameter_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = p['Glp'] / s
    
    tf = (p['Wp'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def dynamics_vis(f = np.linspace(0.05, 2, 100) , p =settings()['parameter_default']):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p['J'] * s**2 - p['mgh'])
    NC = p['Kp'] + p['Kd'] * s
    TD = np.exp(-s * p['dt'])
    F = p['Glp'] / s
    
    tf = (p['Wv'] * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf, f

def objective(theta, keys_fit, fdyn_prop, fdyn_vis, parameter_fix, tf1_exp, tf2_exp):  
    p = dict(zip(keys_fit, theta))
    p.update(parameter_fix)

    tf1_sim, f1 = fdyn_prop(p)
#    err_prop = np.sum(np.log(2 * p['b'] * np.abs(tf1_sim))) + np.sum(np.abs(tf1_sim - tf1_exp) / (p['b'] * np.abs(tf1_sim)))
    err_prop = np.sum(np.abs(tf1_sim - tf1_exp)**2 / (np.abs(tf1_sim))) / len(tf1_sim)
        
    tf2_sim, f2 = fdyn_vis(p)
#    err_vis = np.sum(np.log(2 * p['b'] * np.abs(tf2_sim))) + np.sum(np.abs(tf2_sim - tf2_exp) / (p['b'] * np.abs(tf2_sim)))
    err_vis = np.sum(np.abs(tf2_sim - tf2_exp)**2 / np.abs(tf2_sim)) / len(tf2_sim)
   
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

    tf_sim_prop, f = fdyn_prop(parameter_out)
    tf_sim_vis, f = fdyn_vis(parameter_out)

    return parameter_out, tf_sim_prop, tf_sim_vis, res


def getTFpar(stim, resp, pvn, pv, sr):
    # Obtain threshold bounds
    thresholds = {}
    for m in range(len(pvn)):
        thresholds[f"{pvn[m]}_l"] = pv[m, 1]
        thresholds[f"{pvn[m]}_u"] = pv[m, 2]

    # Get frequency response function
    FD, TD = getFRF(stim, resp, sr, Freqpoints=[1, 2, 4], SmoothFRF=2)
    tf_exp = FD['FRF']
    f = FD['f']
    Coh = FD['Coh']

    # Obtain lookup tables for threshold handling
    Gvth = makeGvth(TD['avg_stim'], thresholds['lambda_l'], thresholds['lambda_u'], sr)
    return f, tf_exp, Coh, Gvth

def makeGvth(xin, lb, ub, sr):
    Gth = np.zeros((101, 13))
    n = 0
    for th in np.linspace(lb, ub, 101):
        xth, _, _ = fd.spectrum(TD_vel_threshold(xin, th, sr), sr)
        xnt, _, _ = fd.spectrum(xin, sr)

        tmp = xth / xnt
        tmp = tmp[::2]  # Take every second element

        tmp_mean = np.array([
            tmp[0],
            np.mean(tmp[0:2]),
            np.mean(tmp[1:3]),
            np.mean(tmp[2:4]),
            np.mean(tmp[3:5]),
            np.mean(tmp[4:6]),
            np.mean(tmp[5:7]),
            np.mean(tmp[6:9]),
            np.mean(tmp[8:11]),
            np.mean(tmp[10:13]),
            np.mean(tmp[12:16]),
            np.mean(tmp[15:20])
        ])
        
        Gth[n, 1:] = tmp_mean
        Gth[n, 0] = th
        n += 1

    return Gth

def getGth(in_val, Gth):
    # Get last entry that is larger than in_val
    ind = np.sum(in_val >= Gth[:, 0]) - 1  # Adjusting for Python's 0-based indexing

    # Estimate values for non-integer indices
    d = (in_val - Gth[ind, 0]) / (Gth[ind + 1, 0] - Gth[ind, 0])
    out = Gth[ind, 1:] * (1 - d) + Gth[ind + 1, 1:] * d
    return out

def TD_vel_threshold(input_signal, threshold, sr):
    # Ensure input is a column vector
    if input_signal.shape[0] < input_signal.shape[1]:
        input_signal = input_signal.T

    # Apply central difference and scale by sampling rate
    input_signal_diff = cdiff(input_signal) * sr

    # Initialize output array
    output = np.zeros_like(input_signal_diff)
    
    # Apply threshold
    threshold_abs = abs(threshold)
    nzp = input_signal_diff > threshold_abs  # Positive values above threshold
    nzn = input_signal_diff < -threshold_abs # Negative values below threshold
    
    output[nzp] = input_signal_diff[nzp] - threshold_abs
    output[nzn] = input_signal_diff[nzn] + threshold_abs

    # Integrate the modified signal
    output = np.cumsum(output) / sr

    # Convert back to original orientation if needed
    if input_signal.shape[0] < input_signal.shape[1]:
        output = output.T

    return output

def cdiff(input_signal):
    # Central difference: computes difference between adjacent elements
    return np.concatenate(([0], np.diff(input_signal)), axis=0)
