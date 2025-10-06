import numpy as np
from numbers import Number
import scipy.signal as signal
import balancepy as bp
import balancepy as bp
from .base_model import BaseModel
import control as control

class Peterka18(BaseModel):
    default_config = {
        "ModelName": 'Peterka 2018',
        "mass_kg": None,
        "height_m": None,
        "data_exp": None
        }

    def _create_parameters(self, mass_kg: Number, height_m: Number):
        """
        Create a set of parameters for the model.
        Args:
            mass_kg (Number): Mass in kilograms.
            height_m (Number): Height in meters.
        Returns:
            ParameterSet: A set of parameters for the model.
        """
        WT = bp.WinterTable(mass_kg, height_m)
        
        mgh = WT.mgh# / 180*np.pi
        J = WT.J# / 180*np.pi
        Kp = 1.35 * WT.mgh# / 180*np.pi
        Kd = 0.47 * WT.mgh# / 180*np.pi

        params = bp.ParameterSet()
        params.add(bp.Parameter("mgh", mgh, bounds=(10, 20), fixed=True))
        params.add(bp.Parameter("J", J, bounds=(0, 0), fixed=True))
        params.add(bp.Parameter("Kp", Kp, bounds=(1.2 * mgh, 1.7 * mgh), fixed=False))
        params.add(bp.Parameter("Kd", Kd, bounds=(0.3 * mgh, 0.77 * mgh), fixed=False))
        params.add(bp.Parameter("W", 0.534, bounds=(0.01, 1), fixed=False))
        params.add(bp.Parameter("dt", 0.151, bounds=(0.1, 0.35), fixed=False))
        params.add(bp.Parameter("Kt", 0.000138, bounds=(0, 0.001), fixed=False))

        return params

    @property
    def dynamics(self):
        r"""
        Dynamics of the Peterka 2018 model. The function returns a transfer function of the model.
        The function is given by

        .. math::

           H_{vis} = \frac{W \cdot C \cdot D \cdot B}{ 1 - F \cdot C \cdot D + C \cdot D \cdot B }

        where :math:`s = i\omega`, :math:`C = K_p + s K_d`, :math:`D = \exp{-s\tau}`, :math:`B = \frac{1}{J\cdot s^2 - mgh}`, and :math:`F = \frac{K_t}{s}`, .

        Reference: Peterka, RJ, Murchison CF, Parrington L, Fino PC, und King LA. Implementation of a Central Sensorimotor Integration Test for Characterization of Human Balance Control During Stance. (2018). https://doi.org/10.3389/fneur.2018.01045.

        Args:
            params_free: Parameters to be optimized.

        Returns:
            err: Objective function value.
        """

        # obtain parameters in dictionary form for easy access
        p = self.params.to_value_dict()

        # Define transfer function as polynomial
        # Target transfer function
        # tf = [(s * W) * NC * TD_num] / [(TD_den * (s * 1/B) - (s * F) * NC * 1/B * TD_num + (s * NC) * TD_num)]
        # Define polynomials
        pade_order = 5
        TD_num, TD_den = control.pade(p['dt'], pade_order)
        NC = [p['Kd'], p['Kp']]
        sNC = [p['Kd'], p['Kp'], 0]
        sF = [p['Kt']]
        invB = [p['J'], 0, -p['mgh']]
        sinvB = [p['J'], 0, -p['mgh'], 0]
        W = p['W']

        # Numerator: W * sNC * TD_num
        num = W * np.convolve(sNC, TD_num)

        # Denominator term 1
        # TD_den * s * 1/B
        den1 = np.convolve(TD_den, sinvB)

        # Denominator term 2
        # sF * NC * 1/B * TD_num
        den2a = sF * np.convolve(NC, invB)
        den2 = np.convolve(den2a, TD_num)

        # Denominator term 3
        # s*NC * TD_num
        den3 = np.convolve(sNC, TD_num)

        # Pad denominator terms to same length
        max_len = max(len(den1), len(den2), len(den3))
        den1p = np.pad(den1, (max_len - len(den1), 0), 'constant')
        den2p = np.pad(den2, (max_len - len(den2), 0), 'constant')
        den3p = np.pad(den3, (max_len - len(den3), 0), 'constant')

        # Combine numerator terms (all have denominator B_poly)
        den12p = np.polyadd(den1p, -den2p)
        den = np.polyadd(den12p, den3p)

        # Regularize small values in the numerator and denominator
        num = [coeff if abs(coeff) > 1e-12 else 1e-12 for coeff in num]
        den = [coeff if abs(coeff) > 1e-12 else 1e-12 for coeff in den]

        transfer_function = signal.TransferFunction(num, den)

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
        assert (self.data_exp is not None 
                and self.data_exp.freq is not None
                and self.data_exp.frf is not None
                ), "No reference data available for fitting"

        # Set parameters if changed e.g. during fitting
        if params_free is not None:
            self.params.set_values(params_free, only_free=True)

        # smooth experimental frequency response function
        frf_exp = self.frf_smoothing(self.data_exp.frf, self.data_exp.freq)

        # method as described in Peterka et al. 2018
        # calculate model frequency response
        f = self.frf_smoothing(self.data_exp.freq, self.data_exp.freq)
        tf = self.dynamics
        w, frf_sim = signal.freqresp(tf, w=f*2*np.pi)

        #calculate objective
        err = np.sum(np.abs(frf_sim - frf_exp) / np.abs(frf_sim))

        return err


    # smoothing function for the frequency response function
    # is applied during the calculation of the objective function
    @staticmethod
    def frf_smoothing(frf, freq):
        index = np.searchsorted(freq, 2.51, side='left')
        if index < 20:
            return _logspace_manual_10s(frf)
        elif index < 75:
            return _logspace_manual_20s(frf)
        else:
            return _logspace_manual_60s(frf)

def _logspace_manual_10s(x):
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


def _logspace_manual_20s(x):
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

def _logspace_manual_60s(x):
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