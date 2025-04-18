import numpy as np
from scipy.optimize import Bounds, basinhopping
from numbers import Number
import scipy.signal as signal
from scipy.signal import convolve as conv
import balancepy as bp
from joblib import Parallel, delayed
import numpy.lib.recfunctions as rfn


from .base_model import balancepyModel

class A23(balancepyModel):


    def __init__(self, mass_kg: Number, height_m: Number):
        WT = bp.WinterTable(mass_kg, height_m)
        
        mgh = WT.mgh / 180*np.pi
        J = WT.J / 180*np.pi
        Kp = 1.3 * WT.mgh / 180*np.pi
        Kd = 0.48 * WT.mgh / 180*np.pi

        self.params = np.array([mgh,    J,      Kp,     Kd,     0.05,    0.17,   0.1, 20,   1])
        self.params_names =        ['mgh',  'J',    'Kp',   'Kd',   'W',    'T',   'Kt', 'Ft', 'b']
        self.parfit_ub = np.array([20, 0, 2*mgh, 1*mgh, 1, 0.3, 0.3, 30, 10])
        self.parfit_lb = np.array([10, 0, mgh, 0, 0.01, 0.05, 0, 3, 0.0001])
        self.parfit_fix_mask = [True, True, False, False, False, False, False, True, False]
        self.transfer_function = A23.get_transfer_function(self.params)
        
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

        self.simulate_FD()


    @staticmethod
    def get_transfer_function(params):
        # implemanted as static method to allow efficient use during bootstrapping
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


    def objective(self, params_free = None, freq = None, reference_frf = None):
        assert (self.FDexp['freq'] is not None or freq is not None), "Please provide a frequency vector for the objective function"
        assert (self.FDexp['frf'] is not None or reference_frf is not None), "Please provide a reference frequency response function for the objective function"

        # Set default parameters
        if params_free is None:
            params = self.params
        else:
            params = self.wrap_params(params_free)

        if freq is None:
            freq = self.FDexp['freq']
        if reference_frf is None:
            reference_frf = self.FDexp['frf']

        assert len(freq) == len(reference_frf), "The lengths of freq and reference_frf must be the same"
        
        #calculate model frequency response
        tf = self.get_transfer_function(params)
        w, frf_sim = signal.freqresp(tf, w=freq*2*np.pi)

        #calculate objective
        err = sum(np.log(2 * params[8] * abs(frf_sim))) + sum(abs(frf_sim - reference_frf) / (params[8] * abs(frf_sim)))

        return err
    
