import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as rfn

def resample(
    time_s: NDArray[np.number],
    data: NDArray[np.number],
    sampling_rate: float,
    end_time_seconds: float = 0,
) -> NDArray:
    """resamples to fixed sample rate

    Args:
        time_s (np.ndarray): 1D timestamps of original recording in seconds
        data (np.ndarray): 1D or 2D data array to be resampled
        sr (int): sampling rate in samples/second
        end_time_seconds (float, optional): optional end time of resampled data in seconds. Defaults to 0.

    Returns:
        NDArray: 1D or 2D with resampled data input
    """

    assert time_s.ndim == 1

    if end_time_seconds == 0:
        end_time_seconds = max(time_s) # get end of recording
        end_time_seconds = end_time_seconds - np.mod(end_time_seconds,1/sampling_rate) # cut at last resampled data point

    n_samples = int(end_time_seconds * sampling_rate)
    new_time_vector = np.linspace(0, end_time_seconds, n_samples, endpoint=False)

    out = interp1d(time_s, data, kind='cubic', fill_value='extrapolate')(new_time_vector)

    return out



def cut_to_cycles(
    data: NDArray,
    cycle_start_samples: int = 0,
    cycle_length_samples: int = 20*90,
    discard_cycles_index: NDArray[int] = []
    ) -> NDArray:
    """
    Cuts the data into cycles based on the provided cycle length, number of cycles, and a given start sample.

    Args:
        data (NDArray): 1D data array to be cut into cycles
        cycle_length_samples (int): Length of cycles in samples
        cycle_start_sample (int, optional): Index of first sample of the first cycle
        discard_cycles_list (NDArray[int], optional): List of cycles to discard starting at 0 for first cycle

    Returns:
        NDArray: 2D array with cycles in columns
    """
    
    assert data.ndim == 1, "Data must be 1D array"

    ncyc = int(np.floor((data.size - cycle_start_samples) / cycle_length_samples))

    # preallocate new matrix
    out = np.empty([cycle_length_samples, ncyc]) 

    for n in range(ncyc):
        i_start = cycle_start_samples + n * cycle_length_samples
        i_end = cycle_start_samples + (n + 1) * cycle_length_samples
        out[:, n] = data[i_start:i_end]

    # remove cycles that are marked to be discarded
    ind = np.ones(ncyc, dtype=bool) # create ncyc-long list of True
    ind[discard_cycles_index] = False # set all discard cycles to False in list

    out = out[:, ind] # select only cycles marked with True

    return out
    


def time_domain_analysis(
        xi: NDArray,
        xo: NDArray,
        samplingrate: int,
        bootstrap_samples: int = 0
) -> NDArray:
    """analyse time domain input/output data.

    Args:
        xi (NDArray): stimulus data
        xo (NDArray): response data
        samplingrate (int): sampling rate in Hz
    """
    
    # Detrend the response data
    xo = np.apply_along_axis(lambda x: x - np.mean(x), 0, xo)

    xi_mean = np.mean(xi,1)
    xo_mean = np.mean(xo,1)

    t = np.arange(1,np.size(xi,0)+1) /samplingrate

    TD = rfn.merge_arrays([
        np.array(t,  dtype=[('time','<f8')]),
        np.array(xi_mean,  dtype=[('stimulus_average','<f8')]),
        np.array(xo_mean,  dtype=[('response_average','<f8')])
        ],
        flatten = True, usemask = False)

    if bootstrap_samples > 0:
        # Calculate confidence intervals using bootstrap
        # This is a placeholder for the actual bootstrap implementation
        # You need to implement the bootstrap logic here
        
        print("Bootstrap confidence intervals are not implemented yet.")

        # from scikits.bootstrap import bootstrap_ci
        # xi_lower, xi_upper = bootstrap_ci(xi, np.mean, n_samples=400)
        # xo_lower, xo_upper = bootstrap_ci(xo, np.mean, n_samples=400)

        # TD = rfn.merge_arrays([
        #     TD,
        #     np.array(xi_lower,  dtype=[('stim_lower','<f8')]),
        #     np.array(xi_upper,  dtype=[('stim_upper','<f8')]),
        #     np.array(xo_lower,  dtype=[('resp_lower','<f8')]),
        #     np.array(xo_upper,  dtype=[('resp_upper','<f8')])
        #     ],
        #     flatten = True, usemask = False)


    # Calculate descriptive parameters of time domain data
    TotalPower = np.mean(xo**2)

    # power of periodic component
    PeriodicPower = np.mean(xo_mean**2)

    # calculation of remnants
    rT = (xo - xo_mean[:, np.newaxis])**2

    # power of remnants
    RemnantPower = np.mean(rT)

    TD_par = {
        'PeriodicPower': PeriodicPower,
        'RemnantPower': RemnantPower,
        'TotalPower': TotalPower
    }

    return TD, TD_par



