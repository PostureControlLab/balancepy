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
    default_config = {
        "ModelName": "Peterka2018",
        "mass_kg": None,
        "height_m": None,
        "stimulus_cycles": None,
        "response_cycles": None,
        "frequencies_Hz": np.arange(0.01, 2.51, 0.01),
        "frequency_selection": 'prts',
        "samplingrate_Hz": 90,
    }

    def create_parameters(self, mass_kg: Number, height_m: Number):
        """
        Create a set of parameters for the model.
        Args:
            mass_kg (Number): Mass in kilograms.
            height_m (Number): Height in meters.
        Returns:
            ParameterSet: A set of parameters for the model.
        """
        WT = bp.WinterTable(mass_kg, height_m)
        
        mgh = WT.mgh / 180*np.pi
        J = WT.J / 180*np.pi
        Kp = 1.45 * WT.mgh / 180*np.pi
        Kd = 0.44 * WT.mgh / 180*np.pi

        params = bp.ParameterSet()
        params.add(bp.Parameter("mgh", mgh, bounds=(10, 20), fixed=True))
        params.add(bp.Parameter("J", J, bounds=(0, 0), fixed=True))
        params.add(bp.Parameter("Kp", Kp, bounds=(1.05* mgh, 2.5 * mgh), fixed=False))
        params.add(bp.Parameter("Kd", Kd, bounds=(0.1*mgh, 1 * mgh), fixed=False))
        params.add(bp.Parameter("W", 0.45, bounds=(0.01, 1), fixed=False))
        params.add(bp.Parameter("dt", 0.16, bounds=(0.1, 0.3), fixed=False))
        params.add(bp.Parameter("Kt", 0.005, bounds=(0, 0.05), fixed=False))

        return params

    def dynamics(self):
        
        p = self.params.to_value_dict()

        num = [ -0.5*p['dt']*p['W']*p['Kd'], (p['W']*p['Kd'] - 0.5*p['W']*p['Kp']*p['dt']), p['W']*p['Kp'], 0 ]

        den = [ (0.5*p['J']*p['dt'] + 0.5*p['Kt']*p['Kd']*p['J']*p['dt'] ), 
                (p['J'] + 0.5*p['Kt']*p['Kp']*p['J']*p['dt'] - p['Kt']*p['Kd']*p['J'] - 0.5*p['Kd']*p['dt']),
                (-0.5*p['mgh']*p['dt'] - p['Kt']*p['Kp']*p['J'] - 0.5*p['Kt']*p['Kd']*p['mgh']*p['dt'] - 0.5*p['Kp']*p['dt'] + p['Kd']),
                (-p['mgh'] - 0.5*p['Kt']*p['Kp']*p['mgh']*p['dt'] + p['Kt']*p['Kd']*p['mgh'] + p['Kp']), 
                p['Kt']*p['Kp']*p['mgh'] ]
        transfer_function = signal.TransferFunction(num, den)

        # Regularize small values in the numerator and denominator
        num = [coeff if abs(coeff) > 1e-12 else 1e-12 for coeff in num]
        den = [coeff if abs(coeff) > 1e-12 else 1e-12 for coeff in den]

        return transfer_function


    def objective(self, params_free = None):
        r"""
        Objective function for optimization.
        Applies a smoothing of the frf across frequencies.
        The error is given by

        .. math::

           \mathrm{err} = \sum_{i} \frac{ \left| H_{\mathrm{sim},i} - H_{\mathrm{exp},i} \right| }{ \left| H_{\mathrm{sim},i} \right| }

        where :math:`H_{\mathrm{sim},i}` is the smoothed simulated FRF at frequency index :math:`i`, and :math:`H_{\mathrm{exp},i}` is the smoothed experimental/reference FRF.

        Args:
            params_free: Parameters to be optimized.

        Returns:
            err: Objective function value.
        """
        assert (self.fit_reference is not None), "No reference data available for fitting"

        # Set parameters if changed e.g. during fitting
        if params_free is not None:
            self.params.set_values(params_free, only_free=True)

        #calculate model frequency response
        tf = self.dynamics()
        w, frf_sim = signal.freqresp(tf, w=self.data_sim.freq*2*np.pi)

        #smooth frequency response functions
        frf_sim = frf_smoothing(frf_sim, self.data_sim.freq)
        frf_exp = frf_smoothing(self.fit_reference, self.data_sim.freq)

        #calculate objective
        err = np.sum(np.abs(frf_sim - frf_exp) / np.abs(frf_sim))

        return err


# smoothing function for the frequency response function
# is applied during the calculation of the objective function
def frf_smoothing(frf, freq):

    index = np.searchsorted(freq, 2.51, side='left')
    if index < 20:
        return logspace_manual_10s(frf)
    elif index < 75:
        return logspace_manual_20s(frf)
    else:
        return logspace_manual_60s(frf)

def logspace_manual_10s(x):
    if x.ndim == 1:
        reduced_x = np.array([
            x[0],               # :,1
            np.mean(x[0:2]),    # :,1:2
            x[1],               # :,2
            np.mean(x[1:3]),    # :,2:3
            np.mean(x[2:4]),    # :,3:4
            np.mean(x[3:5]),    # :,4:5
            np.mean(x[4:7]),    # :,5:7
            np.mean(x[5:9]),    # :,6:9
            np.mean(x[7:10])    # :,8:10
        ])
    elif x.ndim == 2:
        reduced_x = np.array([
            x[:,0],
            np.mean(x[:,0:2], axis=1),
            x[:,1],
            np.mean(x[:,1:3], axis=1),
            np.mean(x[:,2:4], axis=1),
            np.mean(x[:,3:5], axis=1),
            np.mean(x[:,4:7], axis=1),
            np.mean(x[:,5:9], axis=1),
            np.mean(x[:,7:10], axis=1)
        ])
    return reduced_x


def logspace_manual_20s(x):
    if x.ndim == 1:
        reduced_x = np.array([
            x[0],                # :,1
            np.mean(x[0:2]),     # :,1:2
            x[1],                # :,2
            np.mean(x[1:3]),     # :,2:3
            np.mean(x[2:4]),     # :,3:4
            np.mean(x[3:5]),     # :,4:5
            np.mean(x[4:7]),     # :,5:7
            np.mean(x[5:9]),     # :,6:9
            np.mean(x[7:11]),    # :,8:11
            np.mean(x[9:13]),    # :,10:13
            np.mean(x[11:16]),   # :,12:16
            np.mean(x[15:20])    # :,16:20
        ])
    elif x.ndim == 2:
        reduced_x = np.array([
            x[:,0],
            np.mean(x[:,0:2], axis=1),
            x[:,1],
            np.mean(x[:,1:3], axis=1),
            np.mean(x[:,2:4], axis=1),
            np.mean(x[:,3:5], axis=1),
            np.mean(x[:,4:7], axis=1),
            np.mean(x[:,5:9], axis=1),
            np.mean(x[:,7:11], axis=1),
            np.mean(x[:,9:13], axis=1),
            np.mean(x[:,11:16], axis=1),
            np.mean(x[:,15:20], axis=1)
        ])
    return reduced_x

def logspace_manual_60s(x):
    if x.ndim == 1:
        reduced_x = np.array([
            x[0],                 # :,1
            x[1],                 # :,2
            np.mean(x[2:4]),      # :,3:4
            np.mean(x[3:5]),      # :,4:5
            np.mean(x[4:7]),      # :,5:7
            np.mean(x[5:9]),      # :,6:9
            np.mean(x[7:11]),     # :,8:11
            np.mean(x[9:13]),     # :,10:13
            np.mean(x[11:16]),    # :,12:16
            np.mean(x[15:20]),    # :,16:20
            np.mean(x[19:25]),    # :,20:25
            np.mean(x[24:32]),    # :,25:32
            np.mean(x[31:40]),    # :,32:40
            np.mean(x[39:49]),    # :,40:49
            np.mean(x[48:59]),    # :,49:59
            np.mean(x[53:66]),    # :,54:66
            np.mean(x[60:75])     # :,61:75
        ])
    elif x.ndim == 2:
        reduced_x = np.array([
            x[:,0],
            x[:,1],
            np.mean(x[:,2:4], axis=1),
            np.mean(x[:,3:5], axis=1),
            np.mean(x[:,4:7], axis=1),
            np.mean(x[:,5:9], axis=1),
            np.mean(x[:,7:11], axis=1),
            np.mean(x[:,9:13], axis=1),
            np.mean(x[:,11:16], axis=1),
            np.mean(x[:,15:20], axis=1),
            np.mean(x[:,19:25], axis=1),
            np.mean(x[:,24:32], axis=1),
            np.mean(x[:,31:40], axis=1),
            np.mean(x[:,39:49], axis=1),
            np.mean(x[:,48:59], axis=1),
            np.mean(x[:,53:66], axis=1),
            np.mean(x[:,60:75], axis=1)
        ])
    return reduced_x



