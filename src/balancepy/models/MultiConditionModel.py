# from balancepy.models.balancepyModel import balancepyModel
import numpy as np

class MultiConditionModel():
    def __init__(self, *model_list):
        self.model_list = model_list[0]
        self.params_names, \
        self.params, \
        self.ub, \
        self.lb, \
        self.fit_params_fix = self._collect_params()

        # assert somehow that all fixed parameters are the same for all models

    def _collect_params(self):
        M1 = self.model_list[0]

        params_names = []
        parfit_lb = []
        parfit_ub = []
        parfit_fix_mask = []
        params = []

        m=0
        for model in self.model_list:
            self.model_list[m].mcm_paridx = np.zeros(len(model.params_names))

            for idx in range(len(model.params_names)):
                name = model.params_names[idx]

                if name not in params_names:
                    params_names.append(name)
                    parfit_lb = np.append(parfit_lb, model.parfit_lb[idx])
                    parfit_ub = np.append(parfit_ub, model.parfit_ub[idx])
                    parfit_fix_mask = np.append(parfit_fix_mask, model.parfit_fix_mask[idx])
                    params = np.append(params, model.params[idx])
                
                self.model_list[m].mcm_paridx[idx] = params_names.index(name)
            m+=1

        return params_names, params, parfit_ub, parfit_lb, parfit_fix_mask

    def group_objective(self, params):
        
        err = []
        for model in self.model_list:
            theta = params[model.mcm_paridx.astype(int)]
            theta_free, _ = model.unwrap_params(theta)
            err.append(model.objective(theta_free))

        return sum(err)


    def fit(self, reference_frf=None):

        # Set initial guess for free paramss
        theta_free_init = self.unwrap_params()[0]

        # bounds = Bounds(self.parfit_lb[~np.array(self.parfit_fix_mask)], self.parfit_ub[~np.array(self.parfit_fix_mask)])
        bounds = Bounds(self.unwrap_params(self.parfit_lb)[0], self.unwrap_params(self.parfit_ub)[0])
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        if reference_frf is not None:
            objective = lambda theta_free: self.objective(self, theta_free, reference_frf=reference_frf)
            fit_output = basinhopping(objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
        else:
            fit_output = basinhopping(self.objective, theta_free_init, minimizer_kwargs=minimizer_kwargs)
    
        params_fit = self.wrap_params(fit_output.x)

        f = self.FDexp['freq']

        FDsim = self.static_frequency_response(self, params=params_fit, freq=f)
        transfer_function = self.get_transfer_function(params_fit)

        # update class instance, if fit was performed on the experimental data of the instance
        if reference_frf is None:
            self.set_params(params_fit)
            self.FDsim = FDsim
            self.fit_output = fit_output
            self.transfer_function = transfer_function

            self.TDsim = self.simulate(params_fit, np.mean(self.stimulus,1))


        return params_fit, FDsim, transfer_function, fit_output