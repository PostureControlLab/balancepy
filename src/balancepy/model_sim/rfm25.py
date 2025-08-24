import numpy as np
from numpy import convolve as cv
from numbers import Number
import scipy.signal as signal
import balancepy as bp
import balancepy as bp
from .base_model import BaseModel

class RFM25(BaseModel):
    default_config = {
        "ModelName": 'RFM 2025',
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
        
        mgh = WT.mgh / 180*np.pi
        J = WT.J / 180*np.pi
        Kp = 1.4 * WT.mgh / 180*np.pi
        Kd = 0.4 * WT.mgh / 180*np.pi

        params = bp.ParameterSet()
        params.add(bp.Parameter("mgh", mgh, bounds=(10, 20), fixed=True))
        params.add(bp.Parameter("J", J, bounds=(0, 0), fixed=True))
        params.add(bp.Parameter("v_step", 0.3, bounds=(0.3, 0.3), fixed=True))
        params.add(bp.Parameter("Kp", Kp, bounds=(1.05* mgh, 2.5 * mgh), fixed=False))
        params.add(bp.Parameter("Kd", Kd, bounds=(0.1*mgh, 1 * mgh), fixed=False))
        params.add(bp.Parameter("W", 0.45, bounds=(0.01, 1), fixed=False))
        params.add(bp.Parameter("L", 0.3, bounds=(0, 0.8), fixed=False))
        params.add(bp.Parameter("tau", 0.16, bounds=(0.1, 0.3), fixed=False))
        params.add(bp.Parameter("kappa", 0.3, bounds=(0, 1), fixed=False))
        params.add(bp.Parameter("Kt", 0.01, bounds=(0.0001, 0.05), fixed=False))

        return params

    @property
    def dynamics(self):
        r"""
        Dynamics of the Peterka 2018 model. The function returns a transfer function of the model.
        The function is given by

        .. math::

           H_{vis} = \frac{W \cdot NC \cdot TD \cdot B}{ 1 - TF \cdot NC \cdot TD + NC \cdot TD \cdot B }

        where :math:`s = i\omega`, :math:`NC = K_p + s K_d`, :math:`NC = \exp{-s\tau}`, :math:`B = \frac{1}{J\cdot s^2 - mgh}`, and :math:`TF = \frac{K_t}{s}`, .
        """

        p = self.params.to_value_dict()
        
        # Definitions
        chi = (p['v_step'] - p['kappa']) / p['v_step'] if p['v_step'] > p['kappa'] else 0
        W = p['W']
        inv_B = np.array([p['J'], 0, -p['mgh']])

        C = np.array([p['Kd'], p['Kp']])

        inv_F = np.array([1 / p['Kt'], 0])

        Tnum = np.array([-0.5 * p['tau'], 1])
        Tden = np.array([0.5 * p['tau'], 1])


        # Numerator terms
        num1 =       W * cv(cv(inv_F, Tnum), C)  # W * (1/F * Tnum * C)
        num2 = chi * W * cv(cv(inv_F, Tnum), C)  # chi * W * (1/F * Tnum * C)

        num = num1 - num2

        # Denominator terms
        den1 = cv(cv(inv_F, inv_B), Tden)  # (1/F * 1/B * Tden)
        den2 = np.pad(cv(cv(inv_F, Tnum), C), (1, 0))  # 1/F * Tnum * C; # padded to match length
        den3 = cv(cv(inv_B, Tnum), C)     # 1/B * Tnum * C

        den = den1 + den2 - den3  # Combine the denominator terms

        # Regularize small values in the numerator and denominator
        # num = [coeff if abs(coeff) > 1e-12 else 1e-12 for coeff in num]
        # den = [coeff if abs(coeff) > 1e-12 else 1e-12 for coeff in den]

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

        #calculate model frequency response
        tf = self.dynamics
        w, frf_sim = signal.freqresp(tf, w=self.data_exp.freq*2*np.pi)

        #smooth frequency response functions
        frf_sim = self.frf_smoothing(frf_sim, self.data_exp.freq)
        frf_exp = self.frf_smoothing(self.data_exp.frf, self.data_exp.freq)

        #calculate objective
        err = np.sum(np.abs(frf_sim - frf_exp) / np.abs(frf_sim))

        return err



    def approximate_deadzone_tf(self):
        input = self.data_sim.stimulus
        lb, ub = self.params[L].bounds
        samplingrate_Hz = self.data_sim.samplingrate_Hz

        deadzone_tf = []
        n = 0
        for L in np.linspace(lb, ub, 101):
            spec_deadzone = np.fft.fft(self.velocity_deadzone(input, L, samplingrate_Hz))
            spec_no_deadzone = np.fft.fft(input)

            tmp = spec_deadzone[1:] / spec_no_deadzone[1:]

            # Reduce to selected frequencies as defined in data_sim
            tmp = self.data_sim.select_frequencies(tmp)

            deadzone_tf.append(np.concatenate(([L], tmp)))
            n += 1

        return deadzone_tf

    @staticmethod
    def velocity_deadzone(input, kappa, samplingrate_Hz, alpha=0.01):
        """
        Asymmetric threshold function as shown below.
        x is the velocity of the input signal.

        y = (1/2) * sqrt((x - λ)² + α·λ²) - (1/2) * sqrt((x + λ)² + α·λ²) + x
        
        Parameters
        ----------
        input : array_like
            Input signal
        kappa : float
            Threshold parameter λ
        alpha : float
            Smoothing parameter α
            
        Returns
        -------
        y : array_like
            Output signal after applying threshold kappa
        """

        x = np.gradient(input) * samplingrate_Hz  # Assuming sr is defined in the context

        kappa_sq = kappa**2

        term1 = 0.5 * np.sqrt((x - kappa)**2 + alpha * kappa_sq)
        term2 = 0.5 * np.sqrt((x + kappa)**2 + alpha * kappa_sq)

        y = term1 - term2 + x

        output = input[0] + np.cumsum(y) / samplingrate_Hz  # Integrate the output signal

        return output




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