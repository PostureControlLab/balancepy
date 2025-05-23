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
        data_exp (stimulus_response_data or None): Experimental data object.
        data_sim (stimulus_response_data or None): Simulated data object.
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
            mass_kg (Number): Mass in kilograms (optional, can also be provided via config).
            height_m (Number): Height in meters (optional, can also be provided via config).
            config (dict): Configuration dictionary with optional parameters, including mass_kg and height_m.
        """

        # Merge the default configuration with the user-provided config
        config = {**self.default_config, **(config or {})}

        # Allow mass_kg and height_m to be provided via config if not directly passed
        mass_kg = mass_kg if mass_kg is not None else config["mass_kg"]
        height_m = height_m if height_m is not None else config["height_m"]
        if mass_kg is None or height_m is None:
            raise ValueError("Both mass_kg and height_m must be provided.")
        else:
            # Initialize parameters
            self.params = self.create_parameters(mass_kg, height_m)

        self.data_exp = data_exp if data_exp is not None else config["data_exp"]
        self.ModelName = ModelName if ModelName is not None else config["ModelName"]

        self.fit_output = None

        if self.data_exp is not None:
            self.fit()
        else:
            self.freqresp(freq=config["freq"])

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


    def add_data(self,data_exp):

        self.data_exp = data_exp

        self.fit()
        

    def freqresp(self,freq=None):
        
        if freq is not None:
            self.freq = freq
        elif self.data_exp is not None and self.data_exp.freq is not None:
            freq = self.data_exp.freq
        else:
            freq = np.arange(0.01, 2.5, 0.01)
        
        tf = self.dynamics()
        _, frf_sim = signal.freqresp(tf, w=freq*2*np.pi)

        self.data_sim = bp.stimulus_response_data(frf=frf_sim, freq=freq)

        return self.data_sim

    def run_stimulus(self, stimulus, samplingrate_Hz, frequency_selection=None):
        """
        Simulate the time domain response of the system using the provided stimulus.
        Args:
            stimulus (ndarray): The input stimulus for the simulation.
            samplingrate_Hz (float): The sampling rate in Hz.
            frequency_selection (ndarray): Selects frequencies represented in frequency domain representations.
        Returns:
            data_sim (stimulus_response_data): The simulated time domain response.
        """
        assert stimulus.ndim==1, "Stimulus must be a 1D array"
        
        T = np.arange(0, stimulus.shape[0]) / samplingrate_Hz
        
        # Get the system response
        _, response_sim, _ = signal.lsim(self.dynamics(), U=stimulus, T=T)

        self.data_sim = bp.stimulus_response_data(
            samplingrate_Hz=samplingrate_Hz,
            stimulus=stimulus,
            response=response_sim,
            frequency_selection=frequency_selection
        )

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

        # Set initial guess for free paramss
        theta_free_init = self.params.values(only_free=True)

        # bounds = Bounds(self.parfit_lb[~np.array(self.parfit_fix_mask)], self.parfit_ub[~np.array(self.parfit_fix_mask)])
        bounds = self.params.bounds()
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        self.fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
    
        params_fit = self.fit_output.x

        self.params.set_values(params_fit, only_free=True)

        if self.data_exp.stimulus is not None:
            self.run_stimulus(self.data_exp.stimulus.average, self.data_exp.samplingrate_Hz, self.data_exp.frequency_selection)
        else:
            self.freqresp() 

    
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
        

        