import numpy as np
import balancepy as bp
import scipy.signal as signal
import numpy.lib.recfunctions as rfn
from scipy.optimize import basinhopping
from numbers import Number
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class balancepyModel:
    default_config = {
        "ModelName": None,
        "mass_kg": None,
        "height_m": None,
        "stimulus_cycles": None,
        "response_cycles": None,
        "frequencies_Hz": np.arange(0.01, 2.51, 0.01),
        "frequency_selection": None,
        "samplingrate_Hz": None,
    }

    def __init__(self, mass_kg: Number = None, height_m: Number = None, config=None):
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

        # Initialize parameters
        self.params = self.create_parameters(mass_kg, height_m)

        # Assign attributes from the configuration
        self.ModelName = config["ModelName"]
        self.frequency_selection = config["frequency_selection"]
        self.samplingrate_Hz = config["samplingrate_Hz"]

        # Initialize other attributes
        self.data_sim = bp.simulation_data()
        self.data_sim.samplingrate_Hz = self.samplingrate_Hz
        self.data_sim.freq = config["frequencies_Hz"]
        self.data_exp = None
        
        self.fit_reference = None
        self.fit_output = None

        # Check if stimulus_cycles and response_cycles are provided
        # run functions to add stimulus data and/or experimental data
        if config["stimulus_cycles"] is not None:
            self.add_data(
                samplingrate_Hz=self.samplingrate_Hz,
                stimulus_cycles=config["stimulus_cycles"],
                response_cycles=config["response_cycles"],
                frequency_selection=config["frequency_selection"]
            )

    def __repr__(self):
        """
        Provide a detailed string representation of the balancepyModel object.
        Includes ModelName, ParameterSet, frequencies_Hz, fit_reference, and samplingrate_Hz.
        """
        param_summary = repr(self.params) if hasattr(self, 'params') else "No parameters defined"
        frequencies_summary = f"{len(self.freq)} frequencies from {self.freq[0]} to {self.freq[-1]}" if self.freq is not None else "No frequencies defined"
        fit_reference_summary = "Defined" if self.fit_reference is not None else "Not defined"
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
    
    def add_stimulus(self, stimulus_cycles, frequency_selection=None):
        
        self.data_sim.samplingrate_Hz = self.samplingrate_Hz
        self.data_sim.stimulus = stimulus_cycles
        self.frequency_selection = frequency_selection
        self.time = None
        self.freq = None
        self.stimulus_spectrum = None
        self.response_spectrum = None
        self.frf = None
        self.coherence = None

        self.stimulus_cycles = stimulus_cycles

        # calculate input spectrum
        
        if self.stimulus_cycles.ndim < 2:
            yi, yii, f = bp.spectrum(self.stimulus_cycles, self.samplingrate_Hz)
        else:
            yi, yii, f = bp.spectrum(np.mean(self.stimulus_cycles, axis=1), self.samplingrate_Hz)

        if self.stimulus_frequencies_type is not None:
            self.stimulus_frequencies_index = bp.get_frequency_selection(self.stimulus_frequencies_type, self.samplingrate_Hz, f)
        
        if self.stimulus_frequencies_index is not None:
            yi, yii = yi[self.stimulus_frequencies_index], yii[self.stimulus_frequencies_index]

        yi = np.array(yi, dtype=[('stimulus_amplitude_spectrum', 'complex')])
        yii = np.array(yii, dtype=[('stimulus_power_spectrum', '<f8')])

        self.FDsim = rfn.merge_arrays(
            [np.array(f, dtype=[('freq', '<f8')]), yi, yii], flatten=True, usemask=False
        ).view(np.recarray)

        self.simulate_FD()
        self.simulate_TD()


    def add_data(self,samplingrate_Hz,stimulus_cycles,response_cycles=None):

        data_exp = bp.stimulus_response_data(
            samplingrate_Hz=samplingrate_Hz,
            stimulus=stimulus_cycles,
            response=response_cycles,
            frequency_selection=self.frequency_selection
        )

        self.data_exp = data_exp

        self.fit_reference = self.data_exp.frf

        self.data_sim.samplingrate_Hz = samplingrate_Hz
        self.data_sim.stimulus = data_exp.stimulus.average
        self.data_sim.time = data_exp.time
        self.data_sim.frequency_selection = data_exp.frequency_selection

        self.data_sim.freq = data_exp.freq
        self.data_sim.stimulus_spectrum = data_exp.stimulus_spectrum.average

        self.simulate_FD()
        self.simulate_TD()
        

    def simulate_FD(self):
        
        freq = self.data_sim.freq
        
        tf = self.dynamics()
        w, frf_sim = signal.freqresp(tf, w=freq*2*np.pi)

        self.data_sim.frf = frf_sim
        # self.data_sim.gain = abs(frf_sim)
        # self.data_sim.phase = bp.phase(frf_sim, freq)

        return self.data_sim


    def simulate_TD(self, stimulus=None, samplingrate_Hz=None):
        """
        Simulate the time domain response of the system using the provided stimulus.
        Args:
            stimulus (ndarray): The input stimulus for the simulation.
            samplingrate_Hz (float): The sampling rate in Hz.
        Returns:
            TDsim (ndarray): The simulated time domain response.
        """
        assert self.data_sim.stimulus is not None or stimulus is not None, "No stimulus provided for time domain simulation"
        
        if stimulus is not None:
            assert stimulus.ndim == 1, "Stimulus must be a 1D array"
            self.data_sim.samplingrate_Hz = samplingrate_Hz if samplingrate_Hz is not None else self.samplingrate_Hz
            self.data_sim.stimulus = stimulus
            self.data_sim.time = np.arange(0, stimulus.shape[0]) / self.samplingrate_Hz
        
        # Get the system response
        time, response_sim, x_out = signal.lsim(self.dynamics(), U=self.data_sim.stimulus, T=self.data_sim.time)

        self.data_sim.response = response_sim
        self.simulate_FD

        return self.data_sim

    def objective(self, params_free = None):
        assert (self.fit_reference is not None), "No reference data available for fitting"

        # Set parameters if changed e.g. during fitting
        if params_free is not None:
            params = self.params.set_values(params_free, only_free=True)

        #calculate model frequency response
        w, frf_sim = signal.freqresp(self.dynamics(), w=self.freq*2*np.pi)

        #calculate objective
        err = np.sum( np.abs(frf_sim - self.reference_frf) / np.abs(frf_sim) )

        return err
    

    def fit(self):

        # Set initial guess for free paramss
        theta_free_init = self.params.values(only_free=True)

        # bounds = Bounds(self.parfit_lb[~np.array(self.parfit_fix_mask)], self.parfit_ub[~np.array(self.parfit_fix_mask)])
        bounds = self.params.bounds()
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
    
        params_fit = fit_output.x

        self.params.set_values(params_fit, only_free=True)
        self.simulate_FD()
        self.simulate_TD()

    
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
        

        