# from balancepy.models.balancepyModel import balancepyModel
import numpy as np
import numpy.lib.recfunctions as rfn
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
from balancepy.models.ModelClassDefinition import balancepyModel

class MultiConditionModel(balancepyModel):
    def __init__(self, *model_list):
        self.model_list = model_list[0]
        self.params_names, \
        self.params, \
        self.parfit_ub, \
        self.parfit_lb, \
        self.parfit_fix_mask = self._collect_params()

        self.fit_output = None

    def _collect_params(self):
        M1 = self.model_list[0]

        params_names = []
        parfit_lb = []
        parfit_ub = []
        parfit_fix_mask = []
        params = []

        m=0
        for model in self.model_list:
            self.model_list[m].multimodel_paridx = np.zeros(len(model.params_names))

            for idx in range(len(model.params_names)):
                name = model.params_names[idx]

                if name not in params_names:
                    params_names.append(name)
                    parfit_lb = np.append(parfit_lb, model.parfit_lb[idx])
                    parfit_ub = np.append(parfit_ub, model.parfit_ub[idx])
                    parfit_fix_mask = np.append(parfit_fix_mask, model.parfit_fix_mask[idx]).astype(bool)
                    params = np.append(params, model.params[idx])
                else:
                    idx_name = params_names.index(name)
                    if parfit_lb[idx_name] != model.parfit_lb[idx]:
                        print(f"Warning: Lower bound for parameter '{name}' does not match in all models.")
                    if parfit_ub[idx_name] != model.parfit_ub[idx]:
                        print(f"Warning: Upper bound for parameter '{name}' does not match in all models.")
                    if parfit_fix_mask[idx_name] != model.parfit_fix_mask[idx]:
                        print(f"Warning: Fix condition for parameter '{name}' does not match.")
                        parfit_fix_mask[idx_name] = False
                    if params[idx_name] != model.params[idx]:
                        print(f"Warning: Initial guess for parameter '{name}' does not match in all model definitions. Value of first model is used in all models.")
                        self.model_list[m].params[idx] = params[idx_name]

                self.model_list[m].multimodel_paridx[idx] = params_names.index(name)
            m+=1

        return params_names, params, parfit_ub, parfit_lb, parfit_fix_mask

    def objective(self, params_free):
        
        params = self.wrap_params(params_free)

        err = []
        for model in self.model_list:
            model_params = params[model.multimodel_paridx.astype(int)]
            model_params_free, _ = model.unwrap_params(model_params)
            err.append(model.objective(model_params_free))

        return sum(err)


    def fit(self, reference_frf=None):

        # Set initial guess for free paramss
        params_free_init = self.unwrap_params()[0]

        # bounds = Bounds(self.parfit_lb[~np.array(self.parfit_fix_mask)], self.parfit_ub[~np.array(self.parfit_fix_mask)])
        bounds = Bounds(self.unwrap_params(self.parfit_lb)[0], self.unwrap_params(self.parfit_ub)[0])
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}

        if reference_frf is not None:
            objective = lambda params_free: self.objective(self, params_free, reference_frf=reference_frf)
            fit_output = basinhopping(objective, params_free_init, minimizer_kwargs=minimizer_kwargs)
        else:
            fit_output = basinhopping(self.objective, params_free_init, minimizer_kwargs=minimizer_kwargs)
    
        self.fit_output = fit_output
        params_fit = self.wrap_params(fit_output.x)

        for model in self.model_list:
            model_params = params_fit[model.multimodel_paridx.astype(int)]
            model.set_params(model_params)
            model.frequency_response()
            model.simulate()

        return params_fit, fit_output