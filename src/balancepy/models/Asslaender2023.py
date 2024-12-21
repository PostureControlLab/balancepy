import numpy as np
from scipy.optimize import Bounds, basinhopping
from numbers import Number
import scipy.signal as signal
from scipy.signal import convolve as conv
import balancepy as bp
import balancepy.models.Asslaender2023 as Asslaender2023

class Asslaender2023:
    """
    This is the model as described in Peterka et al. 2018 for visual scene tilt perturbations.
    
    Initialize the model with the anthropometric data of the subject.

    This will create a set of default paramss for the model that do not descibe the subject.
    subject specific paramss can be identified by calling the fit method.

    Args:
        mass_kg (Number): mass of the subject in kg
        height_m (Number): height of the subject in m
    
    methods:
        fit(frequencies: np.array, frf_experiment: np.array) -> Tuple[np.array, np.array, OptimizeResult]:
            Fit the model to the given experimental frequency response function
            Returns the optimized paramss, the simulated frequency response function and the optimization result
                    
        set_params(params: np.array) -> None:
            Set the paramss of the model to the given values

        objective(theta_free: np.array) -> float:
            Calculate the objective function of the model for the given paramss

    staticmethod:
        get_transfer_function(params: np.array) -> signal.TransferFunction:
            Returns the transfer function of the model for the given paramss
    """

    def __init__(self, mass_kg: Number, height_m: Number):
        WT = bp.WinterTable(mass_kg, height_m)
        
        mgh = WT.mgh / 180*np.pi
        J = WT.J / 180*np.pi
        Kp = 1.15 * WT.mgh / 180*np.pi
        Kd = 0.3 * WT.mgh / 180*np.pi


        self.params = np.array([mgh,    J,      Kp,     Kd,     0.2,    0.19,   0.1, 20,   1])
        self.names =        ['mgh',  'J',    'Kp',   'Kd',   'W',    'dt',   'Glp', 'Flp', 'b']
        self.ub = np.array([20, 0, 2*mgh, 1*mgh, 1, 0.3, 0.3, 30, 10])
        self.lb = np.array([10, 0, mgh, 0, 0.01, 0.05, 0, 3, 0.0001])
        self.fixed_params_mask = [True, True, False, False, False, False, False, True, False]
        self.transfer_function = Asslaender2023.get_transfer_function(self.params)
        self.logspacing = lambda x: bp.logspace_manual_20s(x)
        self.frequencies = None
        self.frf_experiment = None
        self.frf_simulation = None
        self.fit_output = None
        self.fitOptions = {
            "bootstrapCI": True,
            "N_bootstraps": 400
            }

    def set_params(self, params):
        self.params = params
        self.transfer_function = Asslaender2023.get_transfer_function(self.params)

    @staticmethod
    def get_transfer_function(params):
        
        G, J, Kp, Kd, W, T, Kt = params

        num = [ -0.5*T*W*Kd, (W*Kd - 0.5*W*Kp*T), W*Kp, 0 ]

        den = [ (0.5*J*T + 0.5*Kt*Kd*J*T ), 
                (J + 0.5*Kt*Kp*J*T - Kt*Kd*J - 0.5*Kd*T),
                (-0.5*G*T - Kt*Kp*J - 0.5*Kt*Kd*G*T - 0.5*Kp*T + Kd),
                (-G - 0.5*Kt*Kp*G*T + Kt*Kd*G + Kp), 
                Kt*Kp*G ]

        transfer_function = signal.TransferFunction(num, den)

        return transfer_function


    def objective(self, theta_free = None):
        assert self.frequencies is not None or self.frf_experiment is not None, "Please provide the frequencies and frequency response function of the experiment"

        # Set default parameters
        if theta_free is None:
            theta = self.params
        else:
            # Reconstruct the full theta by filling in the fixed values
            theta = np.zeros(self.fixed_params_mask.__len__())
            theta[self.fixed_params_mask] = self.params[self.fixed_params_mask]  # Set fixed values
            theta[~np.array(self.fixed_params_mask)] = theta_free  # Set free paramss

        #calculate model frequency response
        tf = Asslaender2023.get_transfer_function(theta)
        w, frf_sim = signal.freqresp(tf, w=self.frequencies*2*np.pi)

        #calculate objective
        err = np.sum(np.log(2 * theta(9) * np.abs(frf_sim))) + np.sum(np.abs(frf_sim - self.frf_experiment) / (theta(9) * np.abs(frf_sim)))

        return err

    def fit(self,frequencies,frf_experiment):

        frf_experiment = bp.logspace_manual_20s(frf_experiment)
        frequencies = bp.logspace_manual_20s(frequencies)

        self.frequencies = frequencies
        self.frf_experiment = frf_experiment

        # Create the reduced vectors for free paramss and corresponding bounds
        theta_free_init = self.params[~np.array(self.fixed_params_mask)]  # Only paramss to be optimized
        theta_fixed = self.params[self.fixed_params_mask]  # Fixed paramss
        
        bounds = Bounds(self.lb[~np.array(self.fixed_params_mask)], self.ub[~np.array(self.fixed_params_mask)])
        
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)

        # Reconstruct the full params vector from the estimated paramss res.x and theta_fixed
        params_fit = np.zeros(self.fixed_params_mask.__len__())
        params_fit[self.fixed_params_mask] = theta_fixed
        params_fit[~np.array(self.fixed_params_mask)] = fit_output.x

        self.set_params(params_fit)
        self.fit_output = fit_output
        w, self.frf_simulation = signal.freqresp(self.transfer_function, self.frequencies*2*np.pi)

        return self.params, self.frf_simulation, self.fit_output
    

