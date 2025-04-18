import numpy as np
from scipy.optimize import Bounds, basinhopping
from numbers import Number
import scipy.signal as signal
from scipy.signal import convolve as conv
import balancepy as bp
from joblib import Parallel, delayed
import numpy.lib.recfunctions as rfn
import balancepy as bp
from .base_model import balancepyModel

class P18(balancepyModel):


    def __init__(self, mass_kg: Number, height_m: Number):
        WT = bp.WinterTable(mass_kg, height_m)
        
        mgh = WT.mgh / 180*np.pi
        J = WT.J / 180*np.pi
        Kp = 1.45 * WT.mgh / 180*np.pi
        Kd = 0.44 * WT.mgh / 180*np.pi

        params = bp.ParameterSet()
        params.add(bp.Parameter("mgh", mgh, bounds=(10, 20), fixed=True))
        params.add(bp.Parameter("J", J, bounds=(0, 0), fixed=True))
        params.add(bp.Parameter("Kp", Kp, bounds=(mgh, 2 * mgh), fixed=False))
        params.add(bp.Parameter("Kd", Kd, bounds=(0, 1 * mgh), fixed=False))
        params.add(bp.Parameter("Wv", 0.45, bounds=(0.01, 1), fixed=False))
        params.add(bp.Parameter("dt", 0.16, bounds=(0.05, 0.3), fixed=False))
        params.add(bp.Parameter("Kt", 0.005, bounds=(0, 0.3), fixed=False))

        self.params = params
        
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


    def transfer_function(self):
        
        p = self.params.to_value_dict()

        num = [ -0.5*p['dt']*p['Wv']*p['Kd'], (p['Wv']*p['Kd'] - 0.5*p['Wv']*p['Kp']*p['dt']), p['Wv']*p['Kp'], 0 ]

        den = [ (0.5*p['J']*p['dt'] + 0.5*p['Kt']*p['Kd']*p['J']*p['dt'] ), 
                (p['J'] + 0.5*p['Kt']*p['Kp']*p['J']*p['dt'] - p['Kt']*p['Kd']*p['J'] - 0.5*p['Kd']*p['dt']),
                (-0.5*p['mgh']*p['dt'] - p['Kt']*p['Kp']*p['J'] - 0.5*p['Kt']*p['Kd']*p['mgh']*p['dt'] - 0.5*p['Kp']*p['dt'] + p['Kd']),
                (-p['mgh'] - 0.5*p['Kt']*p['Kp']*p['mgh']*p['dt'] + p['Kt']*p['Kd']*p['mgh'] + p['Kp']), 
                p['Kt']*p['Kp']*p['mgh'] ]
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
        
