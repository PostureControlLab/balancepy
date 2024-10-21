from cmath import pi
from dataclasses import dataclass
from enum import Enum
from numbers import Number
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

def resample(
    time_s: NDArray[np.number],
    data: NDArray[np.number],
    sampling_rate: float,
    end_time_s: float = 0,
) -> NDArray:
    """resamples to fixed sample rate

    Args:
        time_s (NDArray[np.number]): 1D timestamps of original recording in seconds
        data (NDArray[np.number]): 1D or 2D data array to be resampled
        sr (float): sampling rate in samples/second
        end_time_s (float, optional): optional end time of resampled data in seconds. Defaults to 0.

    Returns:
        NDArray: 1D or 2D with resampled data input
    """

    assert time_s.ndim == 1

    if end_time_s == 0:
        end_time_s = max(time_s) # get end of recording
        end_time_s = end_time_s - np.mod(end_time_s,1/sampling_rate) # cut at last resampled data point

    new_time_vector = np.arange(1/sampling_rate, end_time_s+1/sampling_rate, 1/sampling_rate) # define time vector ]0, t_end]

    out = interp1d(time_s[:,0], data[:,0], kind='cubic', fill_value='extrapolate')(new_time_vector)

    return out



def cut_to_cycles(
    data: NDArray[np.number],
    cycle_length_samples: int,
    cycle_start_samples: int = 0,
    discard_cycles_list: list = []
    ) -> NDArray:
    """
    Cuts the data into cycles based on the provided cycle length, number of cycles, and cycle start.

    Args:
    data (NDArray[np.number]): 1D data array to be cut into cycles
    cycle_length_samples (int): Length of cycles in samples
    cycle_start_sample (int, optional): Index of first sample of the first cycle
    discard_cycles_list (NDArray[np.number], optional): List of cycles to discard

    Returns:
    NDArray: 2D array with cycles in rows
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
    ind[discard_cycles_list] = False # set all discard cycles to False in list

    out = out[:, ind] # select only cycles marked with True

    return out
    


    