import numpy as np
from scipy.optimize import Bounds, basinhopping
from numbers import Number
import scipy.signal as signal
from scipy.signal import convolve as conv
import balancepy as bp
import balancepy.models.Peterka2018 as Peterka2018

class Peterka2018:
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
        Kp = 1.45 * WT.mgh / 180*np.pi
        Kd = 0.44 * WT.mgh / 180*np.pi

        self.params = np.array([mgh,    J,      Kp,     Kd,     0.45,    0.16,   0.005])
        self.names =        ['mgh',  'J',    'Kp',   'Kd',   'W',    'dt',   'Glp']
        self.ub = np.array([20, 0, 2*mgh, 1*mgh, 1, 0.3, 0.3])
        self.lb = np.array([10, 0, mgh, 0, 0.01, 0.05, 0])
        self.fixed_params_mask = [True, True, False, False, False, False, False]
        self.transfer_function = Peterka2018.get_transfer_function(self.params)
        self.logspacing = lambda x: bp.logspace_manual_20s(x)
        self.stimulus = None
        self.response = None
        self.FD_frequencies = None
        self.FD_frf_exp = None
        self.FD_frf_exp_uCb = None
        self.FD_frf_exp_lCb = None
        self.FD_frf_sim = None
        self.FD_frf_sim_uCb = None
        self.FD_frf_sim_lCb = None
        
        self.TD_time = None
        self.TD_stimulus_avg = None
        self.TD_response_exp_avg = None
        self.TD_response_exp_uCb = None
        self.TD_response_exp_lCb = None
        self.TD_response_sim_avg = None
        self.TD_response_sim_uCb = None
        self.TD_response_sim_lCb = None

        self.param_uCb = None
        self.param_lCb = None
        self.fit_output = None
        self.fitOptions = {
            "bootstrapCI": True,
            "N_bootstraps": 400
            }
        self.samplingrate: float = 90
        self.selectfreq_nth: int = 2
        self.selectfreq_start_index: int = 0
        self.selectfreq_max_Hz: float = 2

    def set_params(self, params):
        self.params = params
        self.transfer_function = Peterka2018.get_transfer_function(self.params)

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

# def model(params):
    
#     G, J, Kp, Kd, W, T, Kt = params

#     na = [W, 0]
#     nb = [Kd,Kp]
#     nc = [T**2 / 12, -T/2, 1]

#     num = conv(conv(na, nb, mode='full'), nc, mode='full')

#     d1a = [T**2 / 12, T/2, 1]
#     d1b = [J, 0, -G, 0]
#     d2a = [Kd*Kt, Kp*Kt]
#     d2b = [J, 0, -G]
#     d2c = [T**2 / 12, -T/2, 1]
#     d3a = [Kd, Kp]
#     d3b = [T**2 / 12, -T/2, 1]

#     den1 = conv(d1a, d1b, mode='full') # s**5
#     den2 = conv(conv(d2a, d2b, mode='full'), d2c, mode='full') # s**5
#     den3 = conv(d3a, d3b, mode='full') # s**3
#     den3 = np.pad(den3, (len(den2) - len(den3), 0), 'constant')

#     den = den1 - den2 + den3
#     system = signal.TransferFunction(num, den)

#     return system

    def add_experimental_data(self,stimulus,response,samplingrate):
        
        self.stimulus = stimulus
        self.response = response
        self.samplingrate = samplingrate

        self.TD_response_exp_avg = np.mean(response,1)
        self.TD_stimulus_avg = np.mean(stimulus,1)

        FD = bp.frequency_analysis(
                    self.stimulus, 
                    self.response, 
                    self.samplingrate, 
                    self.selectfreq_nth,
                    self.selectfreq_start_index,
                    self.selectfreq_max_Hz
                    )
    
        if self.logspacing == None:
            self.FD_frequencies = FD['f']
            self.FD_frf_exp = FD['frf']
        else:
            self.FD_frequencies = self.logspacing(FD['f'])
            self.FD_frf_exp = self.logspacing(FD['frf'])
            

    def objective(self, theta_free = None):
        assert self.FD_frequencies is not None or self.FD_frf_exp is not None, "Please provide the frequencies and frequency response function of the experiment"

        # Set default parameters
        if theta_free is None:
            theta = self.params
        else:
            # Reconstruct the full theta by filling in the fixed values
            theta = np.zeros(self.fixed_params_mask.__len__())
            theta[self.fixed_params_mask] = self.params[self.fixed_params_mask]  # Set fixed values
            theta[~np.array(self.fixed_params_mask)] = theta_free  # Set free paramss

        #calculate model frequency response
        tf = Peterka2018.get_transfer_function(theta)
        w, frf_sim = signal.freqresp(tf, w=self.FD_frequencies*2*np.pi)

        #calculate objective
        err = np.sum( np.abs(frf_sim - self.FD_frf_exp) / np.abs(frf_sim) )

        return err


    def bootstrap_ConfidenceBounds(self):
        
        yi, yii, f = bp.getSpec(self.stimulus,self.samplingrate)
        yo, yoo, f = bp.getSpec(self.response,self.samplingrate)

        

    def fit(self):

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
        w, self.FD_frf_sim = signal.freqresp(self.transfer_function, self.FD_frequencies*2*np.pi)

        return self.params, self.FD_frf_sim, self.fit_output




