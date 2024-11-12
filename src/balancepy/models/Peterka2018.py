import numpy as np
from scipy.optimize import Bounds, basinhopping
import balancepy.biomechanics as bm
from balancepy.models.helper import dict2par
from numbers import Number

class Peterka2018:
    """
    This is the model as described in Peterka et al. 2018 for visual scene tilt perturbations.
    
    Initialize the model with the anthropometric data of the subject.

    This will create a set of default parameters for the model that do not descibe the subject.
    subject specific parameters can be identified by calling the fit method.

    Args:
        mass_kg (Number): mass of the subject in kg
        height_m (Number): height of the subject in m
    
    """

    winter_table: bm.WinterTable

    def __init__(self, mass_kg: Number, height_m: Number):
        WT = bm.WinterTable(mass_kg, height_m)
        
        mgh = WT.mgh / 180*np.pi
        J = WT.J / 180*np.pi
        Kp = 1.15 * WT.mgh / 180*np.pi
        Kd = 0.3 * WT.mgh / 180*np.pi

        self.start = np.array([mgh,    J,      Kp,     Kd,     0.2,    0.19,   0.1])
        self.names = np.array(['mgh',  'J',    'Kp',   'Kd',   'W',    'dt',   'Glp'])
        self.ub = np.array([20, 0, 2*mgh, 1*mgh, 1, 0.3, 0.3])
        self.lb = np.array([10, 0, mgh, 0, 0.01, 0.05, 0])
        self.fixed_mask = np.array([True, True, False, False, False, False, False])

#  = np.linspace(0.05, 2, 100) 

def frequency_response(f, p):
    
    s = 1j * f * 2 * np.pi
    
    B = 1 / (p(1) * s**2 - p(2))
    NC = p(3) + p(4) * s
    TD = np.exp(-s * p(6))
    F = p(7) / s
    
    tf = (p(5) * NC * B * TD) / (1 - F * NC * TD + NC * B * TD)
    
    return tf

def objective(theta_free, theta_fixed, frequency_response, experimental_data):

    # Reconstruct the full theta by filling in the fixed values
    theta_full = np.zeros_like(theta)
    theta_full[fixed_mask] = theta_fixed  # Set fixed values
    theta_full[~fixed_mask] = theta_free  # Set free parameters

    tf = frequency_response(theta_full)
    
    err = np.sum( np.abs(tf - experimental_data) / (np.abs(tf)) )

    return err

def fit(self,FD):

    f = FD['f']
    experimental_data = FD['FRF']

    # Define the fixed values for the parameters that are fixed
    theta_fixed = self.theta_start[self.fixed_mask]

    # Create the reduced vector for free parameters
    theta_free_init = self.theta_start[~self.fixed_mask]  # Only parameters to be optimized

    model = lambda theta: frequency_response(f, theta)
    obj = lambda theta_free_init: objective(theta_free_init, model, theta_fixed, experimental_data)

    bounds = Bounds(self.lb[~self.fixed_mask], self.ub[~self.fixed_mask])
    
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
    fit_output = basinhopping(obj, theta_free_init, minimizer_kwargs=minimizer_kwargs)

    # Reconstruct the full parameter vector from the estimated parameters res.x and theta_fixed
    parameter_out = np.zeros_like(self.start)
    parameter_out[self.fixed_mask] = theta_fixed
    parameter_out[~self.fixed_mask] = fit_output.x

    tf_sim = model(parameter_out)

    return parameter_out, tf_sim, fit_output


