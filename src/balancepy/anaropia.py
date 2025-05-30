# this submodule contains functions to analyze the data from anaropia balance experiments
import numpy as np
import balancepy as bp
import balancepy.biomechanics as bm
import balancepy.timeseries as ts
from numpy.typing import NDArray as NDArray



def getdata_anaropia(
    filename: str,
    body_height: float=0,
    resample: bool=True,
    samplingrate_Hz: int = 90, # numer gives desired sampling rate; 0 means no resampling
    stimulus: str='stim_tz',
    cut_to_cycles: bool=True,
    end_time: float = 260,
    cycle_start_samples: int = 20*90,
    cycle_length_samples: int = 20*90
) -> NDArray:
    """
    Access and format data from balance experiments recorded with Anaropia.

    Reads data recorded using the Anaropia virtual-reality application for 
    balance experiments. Calculates stimulus and center of mass (COM) data.

    Parameters
    ----------
    filename : str
        Path and filename to be analyzed.
    body_height : float, optional
        Height of subject in meters.
    resample : bool, optional
        If True, resample data to samplingrate_Hz.
    samplingrate_Hz : int, optional
        Desired sampling rate in Hz.
    stimulus : str, optional
        Name of the stimulus column in the data file.
    cut_to_cycles : bool, optional
        If True, cut data to cycles.
    end_time : float, optional
        End time of the experiment in seconds for resampling.
    cycle_start_samples : int, optional
        Start of the first cycle in samples.
    cycle_length_samples : int, optional
        Cycle length in samples.

    Returns
    -------
    com : NDArray
        Experimental center of mass sway in anterior-posterior direction.
    stim : NDArray
        Stimulus data.
    time : NDArray
        Time data.
    """

    # output_frequencies is a vector with the frequencies for which the FRF is calculated; default is up to 2 Hz
    # in case of the prts stimulus sequence, only every odd frequency point has energy, the even frequencies are zero

    data = np.genfromtxt(filename, delimiter=',', names=True)

    if stimulus in data.dtype.names:
        time = data['time']
        stim = data[stimulus]
        com = bm.get_com(data['sho_tz'], np.mean(data['sho_ty']), data['hip_tz'], np.mean(data['hip_ty']),body_height,True)
    else:    
        time = data['time']
        stim = -data['analog4']
        com = bm.get_com(data['shld_zpos'], np.mean(data['shld_ypos']), data['hip_zpos'], np.mean(data['hip_ypos']),body_height,True)


    if resample == True:
        com = ts.resample(time, com, samplingrate_Hz, end_time)
        stim = ts.resample(time, stim, samplingrate_Hz, end_time)
        time = ts.resample(time, time, samplingrate_Hz, end_time)

    if cut_to_cycles == True:
        com = ts.cut_to_cycles(com, cycle_start_samples, cycle_length_samples)
        stim = ts.cut_to_cycles(stim, cycle_start_samples, cycle_length_samples)
        time = ts.cut_to_cycles(time, cycle_start_samples, cycle_length_samples)
    
    return com, stim, time

def getdata_legacy(
    filename: str,
    body_height: float,
    resample: bool=True,
    cut_to_cycles: bool=True,
    sampling_rate: int = 90,
    end_time: float = 220,
    cycle_start_samples: int = 20*90,
    cycle_length_samples: int = 20*90,
    stimulus: str='stim_pitch'
) -> NDArray:
    """
    Access and format data from balance experiments recorded with Anaropia legacy.

    Reads data recorded using the legacy software version of Anaropia for 
    balance experiments. Calculates stimulus and center of mass (COM) data.

    Parameters
    ----------
    filename : str
        Path and filename to be analyzed.
    body_height : float
        Height of subject in meters.
    resample : bool, optional
        If True, resample data to sampling_rate.
    cut_to_cycles : bool, optional
        If True, cut data to cycles.
    sampling_rate : int, optional
        Desired sampling rate in Hz.
    end_time : float, optional
        End time of the experiment in seconds for resampling.
    cycle_start_samples : int, optional
        Start of the first cycle in samples.
    cycle_length_samples : int, optional
        Cycle length in samples.
    stimulus : str, optional
        Name of the stimulus column in the data file.

    Returns
    -------
    com : NDArray
        Experimental center of mass sway in anterior-posterior direction.
    stim : NDArray
        Stimulus data.
    time : NDArray
        Time data.
    """
    
    data = np.genfromtxt(filename, delimiter=',', names=True)

    time = data['time']
    stim = data[stimulus]
    
    com = bm.get_com(data['shld_zpos'], np.mean(data['shld_ypos']), data['hip_zpos'], np.mean(data['hip_ypos']),body_height,True)

    if resample == True:
        com = ts.resample(time, com, sampling_rate, end_time)
        stim = ts.resample(time, stim, sampling_rate, end_time)
        time = ts.resample(time, time, sampling_rate, end_time)

    if cut_to_cycles == True:
        com = ts.cut_to_cycles(com, cycle_start_samples, cycle_length_samples)
        stim = ts.cut_to_cycles(stim, cycle_start_samples, cycle_length_samples)
        time = ts.cut_to_cycles(time, cycle_start_samples, cycle_length_samples)

    return com, stim, time

