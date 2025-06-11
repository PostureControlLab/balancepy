import numpy as np
import balancepy as bp
import scipy.signal as signal
from scipy.optimize import basinhopping
from numbers import Number
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class BaseModel:
    """
    Base class for balancepy model simulations.

    Provides a framework for parameter management, data handling, simulation, and model fitting.

    Parameters
    ----------
    mass_kg : Number, optional
        Mass of the model in kilograms. Must be provided.
    height_m : Number, optional
        Height of the model in meters. Must be provided.
    data_exp : balancepy.sr_data, optional
        Experimental data object containing frequency response data.
    ModelName : str, optional
        Name of the model. If not provided, defaults to None.
    config : dict, optional
        Configuration dictionary to override default settings. If not provided, defaults to None.
    """
    default_config = {
        "ModelName": None,
        "mass_kg": None,
        "height_m": None,
        "data_exp": None
        }

    def __init__(self, mass_kg: Number = None, height_m: Number = None, data_exp = None, ModelName: str = None, config: dict = None):
        
        # Merge the default configuration with the user-provided config
        config = {**self.default_config, **(config or {})}

        self.ModelName = ModelName if ModelName is not None else config["ModelName"]

        # Resolve mass and height
        self.mass_kg = mass_kg if mass_kg is not None else config["mass_kg"]
        self.height_m = height_m if height_m is not None else config["height_m"]
        assert self.mass_kg is not None and self.height_m is not None, "Both mass_kg and height_m must be provided."

        # Initialize parameters
        self.params = self._create_parameters(self.mass_kg, self.height_m)

        self.data_exp = data_exp if data_exp is not None else config["data_exp"]
        assert isinstance(self.data_exp, (bp.sr_data, type(None))), "data_exp must be a sr_data object or None"

        self._update_data_sim()

        self.fit_output = None

    def __repr__(self):

        if hasattr(self, 'params'):
            param_summary = repr(self.params)  
        else: 
            param_summary = "No parameters defined"
        if (self.data_sim is not None and self.data_sim.freq is not None):
            frequencies_summary = f"{len(self.data_sim.freq)} frequencies from {self.data_sim.freq[0]} to {self.data_sim.freq[-1]}"  
        else: frequencies_summary = "No frequencies defined"
        if (self.data_exp is not None and self.data_exp.frf is not None):
            fit_reference_summary = "Defined as data_exp.frf" 
        else: 
            fit_reference_summary = "Not defined"
        if self.data_sim is not None and self.data_sim.samplingrate_Hz is not None:
            samplingrate_summary = f"{self.data_sim.samplingrate_Hz} Hz" 
        else: 
            samplingrate_summary = "Not defined"

        return (
            f"balancepyModel(ModelName={self.ModelName},\n"
            f"  {param_summary},\n"
            f"  Frequencies={frequencies_summary},\n"
            f"  Fit Reference={fit_reference_summary},\n"
            f"  Sampling Rate={samplingrate_summary})"
        )


    def _create_parameters(self, mass_kg: Number, height_m: Number):

        raise NotImplementedError("Subclasses should implement this method.")
        # Example implementation for a specific model
        # WT = bp.WinterTable(mass_kg, height_m)

        # mgh = WT.mgh / 180 * np.pi
        # J = WT.J / 180 * np.pi
        # params = bp.ParameterSet()
        # params.add(bp.Parameter("mgh", mgh, bounds=(10, 20), fixed=True))
        # params.add(bp.Parameter("J", J, bounds=(0, 0), fixed=True))
        # self.params = params

    @property
    def dynamics(self):
        pass
        

    def frf(self,freq=None):
        """
        Returns the frequency response function (FRF) of the system for given input frequencies.

        Parameters
        ----------
        freq : ndarray, optional
            Frequencies in Hz for which to calculate the FRF. If None, uses the frequencies from
            experimental data or defaults to a predefined range.

        Returns
        -------
        frf : ndarray
            The frequency response function of the system.
        """
        if freq is not None:
            data_sim = bp.sr_data()
            data_sim.freq = freq
            _, data_sim.frf = signal.freqresp(self.dynamics, w=data_sim.freq*2*np.pi)
        elif freq is None and self.data_sim is not None and self.data_sim.frf is not None:
            data_sim = self.data_sim
        else:
            data_sim = bp.sr_data()
            data_sim.freq = np.arange(0.01, 2.5, 0.01)
            _, data_sim.frf = signal.freqresp(self.dynamics, w=data_sim.freq*2*np.pi)

        return data_sim


    def simulate_timedomain(self, stimulus, samplingrate_Hz):
        """
        Simulate the time domain response of the system using the provided stimulus.
        
        Parameters
        ----------
        stimulus : ndarray
            The input stimulus for the simulation. Must be a 1D array.
        samplingrate_Hz : float
            The sampling rate in Hz.

        Returns
        -------
        data_sim : balancepy.sr_data
            An object containing the simulated time domain response, including time, stimulus, response, and sampling rate.
        
        Raises
        ------
        AssertionError
            If `stimulus` is not a 1D array.
        """
        assert stimulus.ndim==1, "Stimulus must be a 1D array"
        
        data_sim = bp.sr_data()
        data_sim.samplingrate_Hz = samplingrate_Hz
        data_sim.stimulus = stimulus

        # Get the system response
        _, data_sim.response, _ = signal.lsim(self.dynamics, U=stimulus, T=data_sim.time)

        return data_sim

    def objective(self, params_free = None):
        """
        Objective function for optimization.
        """
        assert (self.fit_reference is not None), "No reference data available for fitting"

        # Set parameters if changed e.g. during fitting
        if params_free is not None:
            params = self.params.set_values(params_free, only_free=True)

        #calculate model frequency response
        w, frf_sim = signal.freqresp(self.dynamics, w=self.freq*2*np.pi)

        #calculate objective
        err = np.sum( np.abs(frf_sim - self.data_exp.frf) / np.abs(frf_sim) )

        return err
    

    def fit(self, data_exp=None):
        """
        Fit the model parameters to the reference experimental frequency response data.

        Uses the basinhopping algorithm to minimize the objective function defined by the model.
        After fitting, updates `data_sim` with the simulated response using the fitted parameters.

        Parameters
        ----------
        data_exp : balancepy.sr_data, optional
            Experimental data object containing frequency response data to fit to.
            If not provided, uses the object's current `data_exp` attribute.

        Returns
        -------
        None
            Updates the model's parameters and `data_sim` attribute in place.
        """
        if data_exp is not None:
            assert isinstance(data_exp, bp.sr_data), "data_exp must be a balancepy.sr_data object"
            self.data_exp = data_exp

        assert (self.data_exp is not None
                and self.data_exp.freq is not None
                and self.data_exp.frf is not None
                ), "No reference data available for fitting"

        # Set initial guess for free paramss
        theta_free_init = self.params.values(only_free=True)

        # bounds = Bounds(self.parfit_lb[~np.array(self.parfit_fix_mask)], self.parfit_ub[~np.array(self.parfit_fix_mask)])
        bounds = self.params.bounds()
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        self.fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
    
        params_fit = self.fit_output.x

        self.params.set_values(params_fit, only_free=True)

        # Update the simulated data object with the system behavior after fitting
        self._update_data_sim()

    def _update_data_sim(self):
        """ Update the simulated data object with the system behavior after fitting."""

        if self.data_exp is None:
            self.data_sim = None
            return
        else:
            # create data_sim object with system behavior after fitting
            data_sim = bp.sr_data()
            # Assign values from data_exp to data_sim
            data_sim.samplingrate_Hz = self.data_exp.samplingrate_Hz
            data_sim.frequency_selection = self.data_exp.frequency_selection
            data_sim.freq = self.data_exp.freq
            data_sim.stimulus_spectrum = self.data_exp.stimulus_spectrum_mean if self.data_exp.stimulus_spectrum is not None else None
            
            data_sim.stimulus = self.data_exp.stimulus_mean if self.data_exp.stimulus is not None else None

            if data_sim.stimulus is not None:
                # simulate response of the system with fitted parameters
                # Repeat stimulus to run twice and discard the first half to remove transient
                stimulus_double = np.concatenate([data_sim.stimulus, data_sim.stimulus])
                time_double = np.arange(0, stimulus_double.shape[0]) / data_sim.samplingrate_Hz
                _, response_double, _ = signal.lsim(self.dynamics, U=stimulus_double, T=time_double)
                data_sim.response = response_double[stimulus_double.shape[0] // 2:]

                # calculate response spectrum
                response_spectrum,_,_ = bp.spectrum(data_sim.response, data_sim.samplingrate_Hz)
                response_spectrum = data_sim.select_frequencies(response_spectrum)
                data_sim.response_spectrum = response_spectrum

            # get frequency response function from dynamnics with fitted parameters
            _, data_sim.frf = signal.freqresp(self.dynamics, w=data_sim.freq*2*np.pi)
            
            self.data_sim = data_sim
    
    def plot(self):
        """
        Plot experimental and simulated data.
        
        This method generates plots for both experimental and simulated data, if available.
        If experimental data is present, it is plotted first. If simulated data is present,
        it is plotted on the same figure as the experimental data. If either dataset is not
        available, a message is printed to notify the user.

        Returns
        -------
        figure : matplotlib.figure.Figure or None
        """
        figure = None

        if self.data_exp is not None:
            if self.data_exp.name is None: self.data_exp.name = 'Experimental' 
            figure = self.data_exp.plot()
        else:
            print('No experimental data available for plotting')

        if self.data_sim is not None:
           if self.data_sim.name is None: self.data_sim.name = 'Simulated'
           figure = self.data_sim.plot(fig = figure) #, params_names=self.params_names, params=self.params)
        else:
            print('No simulated data available for plotting')

        if figure:
            return figure
        

        