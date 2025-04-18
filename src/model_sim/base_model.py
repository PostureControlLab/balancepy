import numpy as np
import balancepy as bp
import scipy.signal as signal
import numpy.lib.recfunctions as rfn
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
from joblib import Parallel, delayed
from numbers import Number


class balancepyModel:

    def __init__(self, mass_kg: Number, height_m: Number):
        WT = bp.WinterTable(mass_kg, height_m)
        
        mgh = WT.mgh / 180*np.pi
        J = WT.J / 180*np.pi

        self.params = None
        
        self.stimulus = None
        self.response = None

        self.FDexp = None
        self.TDexp = None        
        self.FDsim = None
        self.TDsim = None        

        self.fit_output = None

        self.selected_freq = None
        self.frfSmoothing = None

        self.samplingrate: float = None

        self.simulate_FD()


    def transfer_function(self):
        pass

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
        self.reference_frf = self.FDexp['frf']

        self.simulate_FD()
        

    def simulate_FD(self, freq=None):
        
        if freq is None and self.FDexp is not None and 'freq' in self.FDexp.dtype.names:
            freq = self.FDexp['freq']
        elif freq is None:
            freq = np.arange(0.01, 2.01, 0.01)
        
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
            U = self.stimulus
            T = np.arange(0, self.stimulus.shape[0]) / self.samplingrate
        else:
            U = None
            Warning('No stimulus provided for simulation')
        
        if U is not None:
            # Get the system response
            time, TD_response_sim, x_out = signal.lsim(self.transfer_function(), U=U, T=T)

            TDsim = rfn.merge_arrays([  
                np.array(time,    dtype=[('time','<f8')]),
                np.array(U, dtype=[('stimulus_average','<f8')]),
                np.array(TD_response_sim, dtype=[('response_average','<f8')]),
                ],
                flatten = True, usemask = False)

        # Update the class instance if the simulation was performed on the experimental data of the instance
        if stimulus is None: 
            self.TDsim = TDsim

        return TDsim


    def fit(self, reference_frf=None):

        # Set initial guess for free paramss
        theta_free_init = self.params.values(only_free=True)

        # bounds = Bounds(self.parfit_lb[~np.array(self.parfit_fix_mask)], self.parfit_ub[~np.array(self.parfit_fix_mask)])
        bounds = self.bounds()
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        if reference_frf is not None:
            objective = lambda theta_free: self.objective(self, theta_free, reference_frf=reference_frf)
            fit_output = basinhopping(objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
        else:
            fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
    
        params_fit = fit_output.x
        self.params.set_values(params_fit, only_free=True)

        f = self.FDexp['freq']

        FDsim = self.simulate_FD(self, params=params_fit, freq=f)
        transfer_function = self.get_transfer_function(params_fit)

        # update class instance, if fit was performed on the experimental data of the instance
        if reference_frf is None:
            self.set_params(params_fit)
            self.FDsim = FDsim
            self.fit_output = fit_output
            self.transfer_function = transfer_function

            self.TDsim = self.simulate_TD(params_fit, np.mean(self.stimulus,1))


        return params_fit, FDsim, transfer_function, fit_output


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



    def set_params(self, params):

        if isinstance(params, dict):
            for i, name in enumerate(self.params_names):
                if name in params:
                    self.params[i] = params[name]
                else:
                    raise ValueError(f"Parameter {name} not found in the model")
        elif isinstance(params, np.ndarray):
            self.params = params

        self.transfer_function = self.get_transfer_function(self.params)

    def get_params(self):
        if self.params_lCb is not None:
            out = {
                n: (p, lcb, ucb)
                for n, p, lcb, ucb in zip(self.params_names, np.round(self.params,3), np.round(self.params_lCb,3), np.round(self.params_uCb,3))
            }

        else:
            out = dict(zip(self.params_names, self.params))

        return out

    def wrap_params(self, params_free):
        assert len(params_free) == np.sum(~np.array(self.parfit_fix_mask)), "The length of params_free must match the number specified in parfit_fix_mask"
        
        # Reconstruct the full params vector by filling in the fixed values
        params = np.zeros(len(self.parfit_fix_mask))
        params[np.array(self.parfit_fix_mask)] = self.params[np.array(self.parfit_fix_mask)]  # Set fixed values
        params[~np.array(self.parfit_fix_mask)] = params_free  # Set free params

        return params
    
    def unwrap_params(self, params=None):
        if params is None:
            params = self.params
        # Extract the free params from the full params vector
        params_free = params[~np.array(self.parfit_fix_mask)]
        params_fix = params[np.array(self.parfit_fix_mask)]

        return params_free, params_fix
    
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
            figure.show()
            return figure