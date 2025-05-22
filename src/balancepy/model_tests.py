import numpy as np
import balancepy as bp
from concurrent.futures import ProcessPoolExecutor

# Move this function to the top level
def run_model_simulation(args):
    """
    Run a single model simulation.

    Args:
        args (tuple): A tuple containing (model_class, frequencies, weight, height).

    Returns:
        tuple: set_values, fit_result
    """
    model_class, config, weight, height = args

    # Initialize the model instance dynamically
    model = model_class(weight, height, config=config)

    # Select random parameter values within bounds
    for _, param in model.params.items():
        if not param.fixed:
            lower_bound, upper_bound = param.bounds
            param.value = np.random.uniform(lower_bound, upper_bound)

    set_values = model.params.values(only_free=False)

    # Simulate and fit the model
    FDsim = model.simulate_FD()
    TDsim = model.simulate_TD()

    # Add noise to the simulated data
    eta = create_noise(model, realizations=10)

    # Repeat the input spectrum FDsim['yi'] along axis 1 to match the shape of eta
    yi_repeated = np.tile(FDsim['yi'], (eta.shape[1], 1)).T

    # Repeat FDsim['frf'] along axis 1 to match the shape of eta
    # and add the noise divided by the input spectrum to the simulated data
    frf_noise = np.tile(FDsim['frf'], (eta.shape[1], 1)).T + eta / yi_repeated



    model.fit_reference = FDsim['frf']

    # Set default parameters as initial guess for optimization
    model.params.set_defaults()
    model.fit()

    fit_result = model.params.values(only_free=False)

    return set_values, fit_result


def parameter_recovery(model_class, config, num_simulations=100):
    assert isinstance(model_class, type), "model_class must be a class type"
    
    """
    Parameter Recovery for BalancePy Models
    This function performs parameter recovery for a given model class by simulating
    the model with random parameters and then fitting the model to the simulated data.
    Args:   
        model_class (type): The model class to be used for simulation and fitting.
        config (dict): Configuration dictionary containing model parameters.
        num_simulations (int): Number of simulations to run. Default is 100.
    Returns:
        tuple: A tuple containing:
            - model_params (np.ndarray): Array of model parameters from simulations.
            - fit_results (np.ndarray): Array of fitted parameters from simulations.
            - par_names (list): List of parameter names.
    """

    # Prepare arguments for each simulation
    weights = np.random.randint(50, 100, size=num_simulations)
    heights = np.random.uniform(1.5, 2.0, size=num_simulations)
    args = [
        (model_class, config, weights[i], heights[i])
        for i in range(num_simulations)
    ]


    # Run simulations in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(bp.run_model_simulation, args))

    # Extract the parameter values from the results
    model_params = np.array([result[0] for result in results])
    fit_results = np.array([result[1] for result in results])

    # Get parameter names from a sample model
    sample_model = model_class(50, 1.5)
    par_names = sample_model.params.names()

    return model_params, fit_results, par_names


def create_noise(model, realizations=10):
    """
    Create synthetic noise for the given frequency range.

    Args:
        frequencies (array-like): The frequency range.

    Returns:
        array: Synthetic noise.
    """

    # experimentally derived parameters of the two-slope frequency dependence
    # of the gaussian standard deviation of the frequency response function
    two_slope_parameters_real = [0.02691107, 0.71265186, 2.36278279, 0.36362467]
    two_slope_parameters_imag = [0.03128544, 0.71557223, 1.55385433, 0.29383301]

    scale_real = two_slope_function(model.frequencies_Hz, *two_slope_parameters_real)
    scale_imag = two_slope_function(model.frequencies_Hz, *two_slope_parameters_imag)
    noise_real = np.random.normal(0, scale_real, realizations)
    noise_imag = np.random.normal(0, scale_imag, realizations)

    eta = (noise_real + 1j * noise_imag)

    return eta

def two_slope_function(f, a1, b1, b2, f0):
    # Enforce continuity at f0
    a2 = a1 * f0**(b2 - b1)
    # Create the piecewise function
    y = np.piecewise(
        f,
        [f < f0, f >= f0],
        [lambda x: a1 * x**(-b1), lambda x: a2 * x**(-b2)]
    )
    return y