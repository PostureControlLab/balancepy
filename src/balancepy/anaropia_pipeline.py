import numpy as np
import balancepy as bp
import balancepy.biomechanics as bm
import balancepy.timeseries as ts
import balancepy.frequency as fd
from balancepy.frequency import frequency_response_function as get_frf
from numpy.typing import NDArray as NDArray

def stimulus_response_analysis(
    fname: str, 
    body_height: float, 
    body_mass: float
) -> NDArray:
    """run full analysis pipeline for anaropia data

    Args:
        fname (str): path and filename to be analyzed
        body_height (float): height of subject
        body_mass (float): mass of subject

    Returns:
        NDArray: 2D experimental frequency domoin data
        NDArray: 2D experimental time domoin data
    """

    sampling_rate = 90 # numer gives desired sampling rate; 0 means no resampling
    end_time = 260
    cycle_start_samples = 20*sampling_rate
    cycle_length_samples = 20*sampling_rate

    data = np.genfromtxt(fname, delimiter=',', names=True)

    time_raw = data['time']
    stim_raw = data['stim_tz']
    com_raw = bm.com(data['sho_tz'], np.mean(data['sho_ty']), data['hip_tz'], np.mean(data['hip_ty']),body_height,True)

    com = ts.resample(time_raw, com_raw, sampling_rate, end_time)
    stim = ts.resample(time_raw, stim_raw, sampling_rate, end_time)
    time = ts.resample(time_raw, time_raw, sampling_rate, end_time)

    com_cyc = ts.cut_to_cycles(com, cycle_start_samples, cycle_length_samples)
    stim_cyc = ts.cut_to_cycles(stim, cycle_start_samples, cycle_length_samples)

    FD, TD = get_frf(com_cyc, stim_cyc, sampling_rate)

    return FD, TD