import numpy as np
import balancepy as bp
import scipy.signal as signal
import numpy.lib.recfunctions as rfn
from scipy.optimize import basinhopping
from numbers import Number
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class balancepyModel:
    """
    balancepyModel is a base class for balancepy model simulations, providing a framework for parameter management, data handling, simulation, and model fitting.

    Attributes:
        default_config (dict): Default configuration for the model, including keys such as "ModelName", "mass_kg", "height_m", and "data_exp".
        params (ParameterSet): Model parameters, created by the subclass implementation of create_parameters().
        ModelName (str or None): Name of the model.
        data_exp (sr_data or None): Experimental data object.
        data_sim (sr_data or None): Simulated data object.
        fit_output (object or None): Output of the fitting procedure.

    Methods:
        __init__(self, mass_kg: Number = None, height_m: Number = None, data_exp = None, ModelName: str = None)
            Initializes the balancepyModel with mass, height, and optional experimental data and model name.
        __repr__(self)
            Returns a detailed string representation of the balancepyModel object.
        create_parameters(self, mass_kg: Number, height_m: Number)
            Abstract method to create a set of parameters for the model. Must be implemented by subclasses.
        dynamics(self)
            Abstract method to define the system dynamics. Must be implemented by subclasses.
        add_data(self, data_exp)
            Adds experimental data to the model and triggers fitting.
        freqresp(self)
            Simulates the frequency domain response of the system.
        run_stimulus(self, stimulus, samplingrate_Hz, frequency_selection=None)
            Simulates the time domain response of the system using the provided stimulus.
        objective(self, params_free = None)
            Objective function for model fitting, comparing simulated and reference frequency responses.
        fit(self)
            Fits the model parameters to the reference data using optimization.
        plot(self)
            Plots the experimental and simulated frequency response data.
    """
    default_config = {
        "ModelName": None,
        "mass_kg": None,
        "height_m": None,
        "data_exp": None,
        "freq": None
        }

    def __init__(self, mass_kg: Number = None, height_m: Number = None, data_exp = None, ModelName: str = None, config: dict = None):
        """
        Initialize the balancepyModel with mass and height.
        Args:
            ModelName (str, optional, can also be provided via config): Name of the model.
            mass_kg (Number): Mass in kilograms (optional, can also be provided via config).
            height_m (Number): Height in meters (optional, can also be provided via config).
            data_exp (sr_data, optional, can also be provided via config): Experimental data object.
            config (dict, optional): Configuration dictionary with optional parameters.
        """

        # Merge the default configuration with the user-provided config
        config = {**self.default_config, **(config or {})}

        self.ModelName = ModelName if ModelName is not None else config["ModelName"]

        # Resolve mass and height
        self.mass_kg = mass_kg if mass_kg is not None else config["mass_kg"]
        self.height_m = height_m if height_m is not None else config["height_m"]
        assert mass_kg is not None or height_m is not None, "Both mass_kg and height_m must be provided."

        # Initialize parameters
        self.params = self.create_parameters(mass_kg, height_m)

        self.data_exp = data_exp if data_exp is not None else config["data_exp"]
        assert isinstance(self.data_exp, (bp.sr_data, type(None))), "data_exp must be a sr_data object or None"

        self.fit_output = None

        if self.data_exp is not None and self.data_exp.frf is not None:
            self.fit()
        else:
            self.data_sim = None


    def __repr__(self):
        """
        Provide a detailed string representation of the balancepyModel object.
        Includes ModelName, ParameterSet, frequencies_Hz, fit_reference, and samplingrate_Hz.
        """
        param_summary = repr(self.params) if hasattr(self, 'params') else "No parameters defined"
        frequencies_summary = f"{len(self.data_sim.freq)} frequencies from {self.freq[0]} to {self.freq[-1]}" if self.freq is not None else "No frequencies defined"
        fit_reference_summary = "Defined as data_exp.frf" if self.data_exp.frf is not None else "Not defined"
        samplingrate_summary = f"{self.samplingrate_Hz} Hz" if self.samplingrate_Hz is not None else "Not defined"

        return (
            f"balancepyModel(ModelName={self.ModelName},\n"
            f"  {param_summary},\n"
            f"  Frequencies={frequencies_summary},\n"
            f"  Fit Reference={fit_reference_summary},\n"
            f"  Sampling Rate={samplingrate_summary})"
        )


    def create_parameters(self, mass_kg: Number, height_m: Number):
        """
        Create a set of parameters for the model.
        Args:
            mass_kg (Number): Mass in kilograms.
            height_m (Number): Height in meters.
        Returns:
            ParameterSet: A set of parameters for the model.
        """
        raise NotImplementedError("Subclasses should implement this method.")
        # Example implementation for a specific model
        # WT = bp.WinterTable(mass_kg, height_m)

        # mgh = WT.mgh / 180 * np.pi
        # J = WT.J / 180 * np.pi
        # params = bp.ParameterSet()
        # params.add(bp.Parameter("mgh", mgh, bounds=(10, 20), fixed=True))
        # params.add(bp.Parameter("J", J, bounds=(0, 0), fixed=True))
        # self.params = params

    def dynamics(self):
        pass
        

    def frf(self,freq=None):
        """
        Outputs stimulus response data object with the frequency response function (FRF)
        of the system for given input frequencies.
        If freq is not input, output is data_sim.frf or calculated frf using default freq.
        Args:
            freq (ndarray, optional): Frequencies in Hz for which to calculate the FRF. If None, uses the frequencies from data_exp or defaults to a range.
        Returns:
            frf (ndarray): The frequency response function of the system.
        """
        if freq is not None:
            data_sim = bp.sr_data()
            data_sim.freq, data_sim.frf = signal.freqresp(self.dynamics(), w=freq*2*np.pi)
        if freq is None and self.data_sim is not None and self.data_sim.frf is not None:
            data_sim = self.data_sim.frf
        else:
            freq = np.arange(0.01, 2.5, 0.01)
            data_sim.freq, data_sim.frf = signal.freqresp(self.dynamics(), w=freq*2*np.pi)

        return data_sim


    def simulate_timedomain(self, stimulus, samplingrate_Hz):
        """
        Simulate the time domain response of the system using the provided stimulus.
        Args:
            stimulus (ndarray): The input stimulus for the simulation.
            samplingrate_Hz (float): The sampling rate in Hz.
        Returns:
            data_sim: balancepy.sr_data object with the simulated time domain response.
        """
        assert stimulus.ndim==1, "Stimulus must be a 1D array"
        
        data_sim = bp.sr_data()
        data_sim.time = np.arange(0, stimulus.shape[0]) / samplingrate_Hz
        data_sim.samplingrate_Hz = samplingrate_Hz
        data_sim.stimulus = stimulus

        # Get the system response
        _, data_sim.response, _ = signal.lsim(self.dynamics(), U=stimulus, T=data_sim.time)

        return self.data_sim

    def objective(self, params_free = None):
        assert (self.fit_reference is not None), "No reference data available for fitting"

        # Set parameters if changed e.g. during fitting
        if params_free is not None:
            params = self.params.set_values(params_free, only_free=True)

        #calculate model frequency response
        w, frf_sim = signal.freqresp(self.dynamics(), w=self.freq*2*np.pi)

        #calculate objective
        err = np.sum( np.abs(frf_sim - self.data_exp.frf) / np.abs(frf_sim) )

        return err
    

    def fit(self):
        """
        Fits the model parameters to the reference data_exp.frf using optimization.
        This method uses the basinhopping algorithm to minimize the objective function defined by the model.
        Updates data_sim with the simulated response after fitting.
        """
        # Set initial guess for free paramss
        theta_free_init = self.params.values(only_free=True)

        # bounds = Bounds(self.parfit_lb[~np.array(self.parfit_fix_mask)], self.parfit_ub[~np.array(self.parfit_fix_mask)])
        bounds = self.params.bounds()
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        self.fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
    
        params_fit = self.fit_output.x

        self.params.set_values(params_fit, only_free=True)

        # create data_sim object with system behavior after fitting
        data_sim = bp.sr_data()
        # Assign values from data_exp to data_sim
        data_sim.samplingrate_Hz = self.data_exp.samplingrate_Hz
        data_sim.time = self.data_exp.time
        data_sim.stimulus = self.data_exp.stimulus_mean
        data_sim.frequency_selection = self.data_exp.frequency_selection
        data_sim.freq = self.data_exp.freq
        data_sim.stimulus_spectrum = self.data_exp.stimulus_spectrum_mean
        
        if data_sim.stimulus is not None:
            # simulate response of the system with fitted parameters
            # Repeat stimulus to run twice and discard the first half to remove transient
            stimulus_double = np.concatenate([data_sim.stimulus, data_sim.stimulus])
            time_double = np.arange(0, stimulus_double.shape[0]) / data_sim.samplingrate_Hz
            _, response_double, _ = signal.lsim(self.dynamics(), U=stimulus_double, T=time_double)
            data_sim.response = response_double[stimulus_double.shape[0] // 2:]

            # calculate response spectrum
            response_spectrum,_,_ = bp.spectrum(data_sim.response, data_sim.samplingrate_Hz)
            response_spectrum = data_sim.select_frequencies(response_spectrum)
            data_sim.response_spectrum = response_spectrum

        # get frequency response function from dynamnics with fitted parameters
        _, data_sim.frf = signal.freqresp(self.dynamics(), w=data_sim.freq*2*np.pi)
        
        self.data_sim = data_sim
    
    def plot(self):

        figure = None

        if self.data_exp is not None:
            figure = bp.bode_plot(self.data_exp,line_name='Experimental')
        else:
            print('No experimental data available for plotting')

        if self.data_sim is not None:    
           figure = bp.bode_plot(self.data_sim,fig = figure, line_name='Simulated')#, params_names=self.params_names, params=self.params)
        else:
            print('No simulated data available for plotting')

        if figure:
            return figure
        

        