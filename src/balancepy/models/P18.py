import numpy as np
from scipy.optimize import Bounds, basinhopping
from numbers import Number
import scipy.signal as signal
from scipy.signal import convolve as conv
import balancepy as bp
from joblib import Parallel, delayed
import numpy.lib.recfunctions as rfn
import balancepy as bp
from .ModelClassDefinition import balancepyModel

class P18(balancepyModel):


    def __init__(self, mass_kg: Number, height_m: Number):
        WT = bp.WinterTable(mass_kg, height_m)
        
        mgh = WT.mgh / 180*np.pi
        J = WT.J / 180*np.pi
        Kp = 1.45 * WT.mgh / 180*np.pi
        Kd = 0.44 * WT.mgh / 180*np.pi

        self.params = np.array([mgh,    J,      Kp,     Kd,     0.45,    0.16,   0.005])
        self.params_names =        ['mgh',  'J',    'Kp',   'Kd',   'Wv',    'dt',   'Glp']
        self.parfit_ub = np.array([20, 0, 2*mgh, 1*mgh, 1, 0.3, 0.3])
        self.parfit_lb = np.array([10, 0, mgh, 0, 0.01, 0.05, 0])
        self.parfit_fix_mask = [True, True, False, False, False, False, False]
        self.transfer_function = self.get_transfer_function(self.params)
        
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

        self.frequency_response()


    @staticmethod
    def get_transfer_function(params):
        
        G, J, Kp, Kd, Wv, T, Kt = params

        num = [ -0.5*T*Wv*Kd, (Wv*Kd - 0.5*Wv*Kp*T), Wv*Kp, 0 ]

        den = [ (0.5*J*T + 0.5*Kt*Kd*J*T ), 
                (J + 0.5*Kt*Kp*J*T - Kt*Kd*J - 0.5*Kd*T),
                (-0.5*G*T - Kt*Kp*J - 0.5*Kt*Kd*G*T - 0.5*Kp*T + Kd),
                (-G - 0.5*Kt*Kp*G*T + Kt*Kd*G + Kp), 
                Kt*Kp*G ]

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
        err = np.sum( np.abs(frf_sim - reference_frf) / np.abs(frf_sim) )

        return err
        
# attempt to formulate model using convolutions to allow easier implementation of 
# new model components
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