import numpy as np
import balancepy as bp
import scipy.signal as signal
import numpy.lib.recfunctions as rfn
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
from joblib import Parallel, delayed
from numbers import Number
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class balancepyModel:
    default_config = {
        "ModelName": None,
        "stimulus": None,
        "response": None,
        "FDexp": None,
        "TDexp": None,
        "FDsim": None,
        "TDsim": None,
        "frequencies": np.arange(0.01, 2.01, 0.01),
        "fit_reference": None,
        "fit_output": None,
        "params_uCb": None,
        "params_lCb": None,
        "selected_freq": None,
        "frfSmoothing": None,
        "samplingrate": None,
    }

    def __init__(self, mass_kg: Number, height_m: Number, config=None):
        """
        Initialize the balancepyModel with mass and height.
        Parameters:
            mass_kg (Number): Mass in kilograms.
            height_m (Number): Height in meters.
            config (dict): Configuration dictionary with optional parameters.
        """

        # Initialize parameters
        self.params = self.create_parameters(mass_kg, height_m)


        # Merge the default configuration with the user-provided config
        config = {**self.default_config, **(config or {})}

        # Assign attributes from the configuration
        self.ModelName = config["ModelName"]
        self.stimulus = config["stimulus"]
        self.response = config["response"]
        self.FDexp = config["FDexp"]
        self.TDexp = config["TDexp"]
        self.FDsim = config["FDsim"]
        self.TDsim = config["TDsim"]
        self.frequencies = config["frequencies"]
        self.fit_reference = config["fit_reference"]
        self.fit_output = config["fit_output"]
        self.params_uCb = config["params_uCb"]
        self.params_lCb = config["params_lCb"]
        self.selected_freq = config["selected_freq"]
        self.frfSmoothing = config["frfSmoothing"]
        self.samplingrate = config["samplingrate"]

        # Perform initial simulation
        self.simulate_FD()
        self.simulate_TD()

    def __repr__(self):
        """
        Provide a detailed string representation of the balancepyModel object.
        Includes ModelName, ParameterSet, frequencies, fit_reference, frfSmoothing, and samplingrate.
        """
        param_summary = repr(self.params) if hasattr(self, 'params') else "No parameters defined"
        frequencies_summary = f"{len(self.frequencies)} frequencies from {self.frequencies[0]} to {self.frequencies[-1]}" if self.frequencies is not None else "No frequencies defined"
        fit_reference_summary = "Defined" if self.fit_reference is not None else "Not defined"
        frf_smoothing_summary = repr(self.frfSmoothing) if self.frfSmoothing is not None else "Not defined"
        samplingrate_summary = f"{self.samplingrate} Hz" if self.samplingrate is not None else "Not defined"

        return (
            f"balancepyModel(ModelName={self.ModelName},\n"
            f"  {param_summary},\n"
            f"  Frequencies={frequencies_summary},\n"
            f"  Fit Reference={fit_reference_summary},\n"
            f"  FRF Smoothing={frf_smoothing_summary},\n"
            f"  Sampling Rate={samplingrate_summary})"
        )


    def create_parameters(self, mass_kg: Number, height_m: Number):
        """
        Create a set of parameters for the model.
        Parameters:
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

    def transfer_function(self):
        pass
    
    def add_experimental_data(self,stimulus,response,samplingrate):
        
        self.stimulus = stimulus
        self.response = response
        self.samplingrate = samplingrate

        time = np.arange(0, stimulus.shape[0]) / samplingrate
        self.TDexp = rfn.merge_arrays([
                    np.array(time,    dtype=[('time','<f8')]),
                    np.array(np.mean(stimulus,1), dtype=[('stimulus_average','<f8')]),
                    np.array(np.mean(response,1), dtype=[('response_average','<f8')])
                    ])  

        self.FDexp = bp.frequency_analysis(
                    self.stimulus, 
                    self.response, 
                    self.samplingrate, 
                    self.selected_freq,
                    self.frfSmoothing,
                    )
        
        self.frequencies = self.FDexp['freq']
        self.fit_reference = self.FDexp['frf']

        self.simulate_FD()
        

    def simulate_FD(self):
        
        freq = self.frequencies
        
        tf = self.transfer_function()
        w, frf_sim = signal.freqresp(tf, w=freq*2*np.pi)

        FDsim = rfn.merge_arrays([
            np.array(freq,    dtype=[('freq','<f8')]),
            np.array(frf_sim, dtype=[('frf','complex')]),
            np.array(np.abs(frf_sim), dtype=[('gain','<f8')]),
            np.array(bp.phase(frf_sim,freq),  dtype=[('phase','<f8')])
            ],
            flatten = True, usemask = False)

        # Update the class instance if the simulation was performed on the experimental data of the instance
        self.FDsim = FDsim

        return FDsim


    def simulate_TD(self, stimulus=None):
        
        if stimulus is not None:
            self.stimulus = stimulus
            U = stimulus
            T = np.arange(0, stimulus.shape[0]) / self.samplingrate
        elif stimulus is None and self.TDexp is not None and 'stimulus_average' in self.TDexp.dtype.names:
            U = self.TDexp['stimulus_average']
            T = self.TDexp['time']
        elif stimulus is None and self.stimulus is not None:
            U = np.mean(self.stimulus, axis=1) if self.stimulus.ndim > 1 else self.stimulus
            T = np.arange(0, self.stimulus.shape[0]) / self.samplingrate
        else:
            U = None
            # Warning('No stimulus provided for simulation')
        
        if U is not None:
            # Get the system response
            time, TD_response_sim, x_out = signal.lsim(self.transfer_function(), U=U, T=T)

            # for better comparability data is centered around 0
            response_average = TD_response_sim - np.mean(TD_response_sim)

            TDsim = rfn.merge_arrays([  
                np.array(time,    dtype=[('time','<f8')]),
                np.array(U, dtype=[('stimulus_average','<f8')]),
                np.array(response_average, dtype=[('response_average','<f8')]),
                ],
                flatten = True, usemask = False)

            # Update the class instance if the simulation was performed on the experimental data of the instance
            if stimulus is None: 
                self.TDsim = TDsim

            return TDsim

    def objective(self, params_free = None):
        assert (self.frequencies is not None), "Please provide a frequency vector for the objective function"
        assert (self.reference_frf is not None), "Please provide a reference frequency response function for the objective function"
        assert len(self.frequencies) == len(self.reference_frf), "The lengths of frequencies and reference_frf vectors must be the same"

        # Set parameters if changed e.g. during fitting
        if params_free is not None:
            params = self.params.set_values(params_free, only_free=True)

        #calculate model frequency response
        w, frf_sim = signal.freqresp(self.transfer_function(), w=self.frequencies*2*np.pi)

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



    def ConfidenceBounds_fit(self, N_bootstraps = 200):
        # get spetctra of stimulus and response
        yi, _, f = bp.spectrum(self.stimulus,self.samplingrate)
        yo, _, f = bp.spectrum(self.response,self.samplingrate)

        # handle frequency selection
        if isinstance(self.selected_freq, np.ndarray):
            selected_freq = self.selected_freq
        elif self.selected_freq == 'prts': # selects every second frequency point up to 2 Hz
            selected_freq = np.arange(
                0,
                int(round(2 * np.size(self.response, 0) / self.samplingrate)), 
                2
            )
        elif self.selected_freq == 'all':
            selected_freq = np.arange(0, np.size(f), 1)

        f   = f[selected_freq]
        yi  = yi[selected_freq,:]
        yo  = yo[selected_freq,:]

        # perform bootstraps for frequency response function
        n_samples = yo.shape[0]
        bootstrap_indices = [
            np.random.choice(n_samples, size=n_samples, replace=True)
            for _ in range(N_bootstraps)
        ]

        bootstrap_frf = Parallel(n_jobs=-1)(
            delayed(bp.frf)(
                yi[idx], yo[idx]
            ) for idx in bootstrap_indices
        )
        
        bootstrap_smoothfrf = np.array([self.frfSmoothing(frf,f) for frf in bootstrap_frf])

        bootstrap_fit_results = Parallel(n_jobs=-1)(
            delayed(self.fit)(reference_frf=frf) for frf in bootstrap_smoothfrf
        )

        bootstrap_params = []
        for result in bootstrap_fit_results:
            bootstrap_params.append(result[0])

        # Extract upper and lower confidence bounds from bootstrap results
        self.params_uCb = np.percentile(bootstrap_params, 97.5, axis=0)
        self.params_lCb = np.percentile(bootstrap_params, 2.5, axis=0)
        
        return self.params_uCb, self.params_lCb

    
    def plot(self):

        figure = None

        if self.FDexp is not None:
            figure = bp.bode_plot(self.FDexp, self.TDexp,line_name='Experimental')
        else:
            print('No experimental data available for plotting')

        if self.FDsim is not None:    
           figure = bp.bode_plot(self.FDsim, self.TDsim,fig = figure, line_name='Simulated')#, params_names=self.params_names, params=self.params)
        else:
            print('No simulated data available for plotting')

        if figure:
            return figure