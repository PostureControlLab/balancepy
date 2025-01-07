import numpy as np
from scipy.optimize import Bounds, basinhopping
from numbers import Number
import scipy.signal as signal
from scipy.signal import convolve as conv
import balancepy as bp
from joblib import Parallel, delayed
import numpy.lib.recfunctions as rfn

import balancepy.models.Asslaender2023 as Asslaender2023

class Asslaender2023:
    """
    This is the model as described in Asslaender et al. 2023 for visual scene tilt perturbations.
    
    Initialize the model with the anthropometric data of the subject.

    This will create a set of default paramss for the model that do not descibe the subject.
    subject specific paramss can be identified by calling the fit method.

    Args:
        mass_kg (Number): mass of the subject in kg
        height_m (Number): height of the subject in m
    
    methods:
        fit(freq: np.array, frf_experiment: np.array) -> Tuple[np.array, np.array, OptimizeResult]:
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
        self.names =        ['mgh',  'J',    'Kp',   'Kd',   'W',    'T',   'Kt', 'Ft', 'b']
        self.ub = np.array([20, 0, 2*mgh, 1*mgh, 1, 0.3, 0.3, 30, 10])
        self.lb = np.array([10, 0, mgh, 0, 0.01, 0.05, 0, 3, 0.0001])
        self.fixed_params_mask = [True, True, False, False, False, False, False, True, False]
        self.transfer_function = Asslaender2023.get_transfer_function(self.params)
        
        self.stimulus = None
        self.response = None

        self.FDexp = None
        self.TDexp = None        
        self.FDsim = None
        self.TDsim = None        

        self.params_uCb = None
        self.params_lCb = None
        self.fit_output = None

        self.selected_freq = 'prts'
        self.frfSmoothing = lambda x, f: bp.logspace_manual_20s(x,f)

        self.samplingrate: float = 90


    def set_params(self, params):
        self.params = params
        self.transfer_function = Asslaender2023.get_transfer_function(self.params)

    @staticmethod
    def get_transfer_function(params):
        
        G, J, Kp, Kd, W, T, Kt, Ft, b = params

        num = [ -0.5*T*W*Kd*Ft, 
               (W*Kd*Ft - 0.5*W*Kp*Ft*T - 0.5*T*W*Kd), 
               W*Kp*Ft + W*Kd - 0.5*W*Kp*T, 
               W*Kp ]

        den = [ (0.5*Ft*J*T + 0.5*Kt*Kd*J*T ), 
                (Ft*J + 0.5*Kt*Kp*J*T - Kt*Kd*J - 0.5*Kd*T),
                (-0.5*G*Ft*T + J - Kt*Kp*J - 0.5*Kt*Kd*G*T - 0.5*Kp*Ft*T + Kd*Ft - 0.5*Kd*T),
                (-Ft*G -0.5*G*T - 0.5*Kt*Kp*G*T + Kt*Kd*G + Kp*Ft - 0.5*Kp*T + Kd), 
                (-G + Kt*Kp*G + Kp) ]

        transfer_function = signal.TransferFunction(num, den)

        return transfer_function

    def add_experimental_data(self,stimulus,response,samplingrate):
        
        self.stimulus = stimulus
        self.response = response
        self.samplingrate = samplingrate

        time = np.arange(0, stimulus.shape[0]) / samplingrate
        self.TDexp = rfn.merge_arrays([
                    np.array(time,    dtype=[('time','<f8')]),
                    np.array(np.mean(stimulus,1), dtype=[('stimulus_avg','<f8')]),
                    np.array(np.mean(response,1), dtype=[('response_avg','<f8')])
                    ])  

        self.FDexp = bp.frequency_analysis(
                    self.stimulus, 
                    self.response, 
                    self.samplingrate, 
                    self.selected_freq,
                    self.frfSmoothing,
                    )


    def objective(self, theta_free = None, freq = None, reference_frf = None):
        assert (self.FDexp['freq'] is not None or freq is not None), "Please provide a frequency vector for the objective function"
        assert (self.FDexp['frf'] is not None or reference_frf is not None), "Please provide a reference frequency response function for the objective function"

        # Set default parameters
        if theta_free is None:
            theta = self.params
        else:
            # Reconstruct the full theta by filling in the fixed values
            theta = np.zeros(self.fixed_params_mask.__len__())
            theta[self.fixed_params_mask] = self.params[self.fixed_params_mask]  # Set fixed values
            theta[~np.array(self.fixed_params_mask)] = theta_free  # Set free paramss

        if freq is None:
            freq = self.FDexp['freq']
        if reference_frf is None:
            reference_frf = self.FDexp['frf']

        assert len(freq) == len(reference_frf), "The lengths of freq and reference_frf must be the same"
        
        #calculate model frequency response
        tf = Asslaender2023.get_transfer_function(theta)
        w, frf_sim = signal.freqresp(tf, w=freq*2*np.pi)

        #calculate objective
        err = np.sum(np.log(2 * theta[8] * np.abs(frf_sim))) + np.sum(np.abs(frf_sim - reference_frf) / (theta[8] * np.abs(frf_sim)))

        return err

        

    def simulate(self, params=None, stimulus=None):
        assert (params is not None or self.params is not None), "Please provide the parameters for the simulation"
        assert (stimulus is not None or self.stimulus is not None), "Please provide the stimulus for the simulation"

        if params is None:
            params = self.params
        if stimulus is None:
            U = self.stimulus
        else:
            U = stimulus

        time = np.arange(0, stimulus.shape[0]) / self.samplingrate

        # Get the system response
        time, TD_response_sim, x_out = signal.lsim(self.transfer_function, U=U, T=time)

        if stimulus is not None:
            self.TD_time = time
            self.TD_response_sim_avg = TD_response_sim

        return TD_response_sim, time

    def fit(self, reference_frf=None):

        # Create the reduced vectors for free paramss and corresponding bounds
        theta_free_init = self.params[~np.array(self.fixed_params_mask)]  # Only paramss to be optimized
        theta_fixed = self.params[self.fixed_params_mask]  # Fixed paramss
        
        bounds = Bounds(self.lb[~np.array(self.fixed_params_mask)], self.ub[~np.array(self.fixed_params_mask)])
        
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        if reference_frf is not None:
            objective = lambda theta_free: Asslaender2023.objective(self, theta_free, reference_frf=reference_frf)
            fit_output = basinhopping(objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
        else:
            fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
    

        # Reconstruct the full params vector from the estimated paramss res.x and theta_fixed
        params_fit = np.zeros(self.fixed_params_mask.__len__())
        params_fit[self.fixed_params_mask] = theta_fixed
        params_fit[~np.array(self.fixed_params_mask)] = fit_output.x
        f = self.FDexp['freq']

        w, frf_sim = signal.freqresp(self.transfer_function, f*2*np.pi)

        # update class instance, if fit was performed on the experimental data of the instance
        if reference_frf is None:
            response_sim, time = self.simulate(params_fit, np.mean(self.stimulus,1))

            self.set_params(params_fit)
            self.fit_output = fit_output

            self.FDsim = rfn.merge_arrays([
                np.array(f,    dtype=[('freq','<f8')]),
                np.array(frf_sim, dtype=[('frf','complex')]),
                np.array(np.abs(frf_sim), dtype=[('gain','<f8')]),
                np.array(bp.phase(frf_sim,f),  dtype=[('phase','<f8')])
                ],
                flatten = True, usemask = False)

            self.TDsim = rfn.merge_arrays([  
                np.array(time,    dtype=[('time','<f8')]),
                np.array(response_sim, dtype=[('response','complex')]),
                ],
                flatten = True, usemask = False)

        return params_fit, frf_sim, fit_output


    def ConfidenceBounds_fit(self, N_bootstraps = 200):
        # get spetctra of stimulus and response
        yi, yii, f = bp.spectrum(self.stimulus,self.samplingrate)
        yo, yoo, f = bp.spectrum(self.response,self.samplingrate)

        # handle frequency selection
        if isinstance(self.selected_freq, np.ndarray):
            selected_freq = self.selected_freq
        elif self.selected_freq == 'prts': # selects every second frequency point up to 2 Hz
            selected_freq = np.arange(
                0,
                int(round(2 * np.size(self.response, 0) / self.samplingrate)), 
                2
            )
        elif self.selected_freq == 'all':
            selected_freq = np.arange(0, np.size(f), 1)

        f   = f[selected_freq]
        yi  = yi[selected_freq,:]
        yo  = yo[selected_freq,:]

        # perform bootstraps for frequency response function
        n_samples = yo.shape[0]
        bootstrap_indices = [
            np.random.choice(n_samples, size=n_samples, replace=True)
            for _ in range(N_bootstraps)
        ]

        bootstrap_frf = Parallel(n_jobs=-1)(
            delayed(bp.frf)(
                yi[idx], yo[idx]
            ) for idx in bootstrap_indices
        )
        
        bootstrap_smoothfrf = np.array([self.frfSmoothing(frf,f) for frf in bootstrap_frf])

        bootstrap_fit_results = Parallel(n_jobs=-1)(
            delayed(self.fit)(reference_frf=frf) for frf in bootstrap_smoothfrf
        )

        bootstrap_params = []
        for result in bootstrap_fit_results:
            bootstrap_params.append(result[0])

        # Extract upper and lower confidence bounds from bootstrap results
        self.params_uCb = np.percentile(bootstrap_params, 97.5, axis=0)
        self.params_lCb = np.percentile(bootstrap_params, 2.5, axis=0)
        
        return self.params_uCb, self.params_lCb
