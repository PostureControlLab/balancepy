# this submodule contains functions to analyze the data from anaropia balance experiments
# 

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

def _getdata_lifespan(
    filename: str,
    body_height: float,
    resample: bool=True,
    cut_to_cycles: bool=True,
    sampling_rate: int = 90,
    end_time: float = 220,
    cycle_start_samples: int = 20*90,
    cycle_length_samples: int = 20*90
) -> NDArray:
    """
    Access and format data from the lifespan dataset.

    Reads data recorded from lifespan balance experiments and returns the time
    domain data for the center of mass (COM), visual stimulus, and 
    proprioceptive stimulus. Note that for this dataset, surface tilts and
    visual scene tilts were applied.

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

    Returns
    -------
    com : NDArray
        Experimental center of mass sway in anterior-posterior direction.
    stim_vis : NDArray
        Visual stimulus data.
    stim_prop : NDArray
        Proprioceptive stimulus data.
    time : NDArray
        Time data.
    """

    data = np.genfromtxt(filename, delimiter=',', names=True)

    time = data['time']
    stim_vis = data['stim_pitch']
    stim_surf = -data['analog4']
    
    com = bm.get_com(data['shld_zpos'], np.mean(data['shld_ypos']), data['hip_zpos'], np.mean(data['hip_ypos']),body_height,True)

    if resample == True:
        com = ts.resample(time, com, sampling_rate, end_time)
        stim_vis = ts.resample(time, stim_vis, sampling_rate, end_time)
        stim_prop = ts.resample(time, stim_surf, sampling_rate, end_time)
        time = ts.resample(time, time, sampling_rate, end_time)

    if cut_to_cycles == True:
        com = ts.cut_to_cycles(com, cycle_start_samples, cycle_length_samples)
        stim_vis = ts.cut_to_cycles(stim_vis, cycle_start_samples, cycle_length_samples)
        stim_prop = ts.cut_to_cycles(stim_prop, cycle_start_samples, cycle_length_samples)
        time = ts.cut_to_cycles(time, cycle_start_samples, cycle_length_samples)

    return com, stim_vis, stim_prop, time

# def analysis_prts(
#     xi: NDArray,
#     xo: NDArray,
#     samplingrate: float,
#     selected_frequencies: NDArray = 0,
#     smoothPhase: bool=True,
#     logspace: bool=True,
# ) -> NDArray:
    
#     yi, yii, f = fd.spectrum(xi, samplingrate)
#     yo, yoo, f = fd.spectrum(xo, samplingrate)

#     if selected_frequencies == 0:
#     selected_frequencies_prop = range(0, int(2 * np.size(com_cyc, 0) / sampling_rate), 2)
#     def get_frf_prop(stim_prop, com, sampling_rate, selected_frequencies_prop):
#         frf_prop = frequency_analysis(stim_prop_cyc, com_cyc, sampling_rate, selected_frequencies_prop)
#         return frf_prop

#     selected_frequencies_vis = range(1, int(2 * np.size(com_cyc, 0) / sampling_rate), 4)
#     def get_frf_vis(stim_prop, com, sampling_rate, selected_frequencies_vis):
#         frf_vis = frequency_analysis(stim_vis_cyc, com_cyc, sampling_rate, selected_frequencies_vis)
#         return frf_vis


# def analysis_lifespan(
#         fname: str, 
#         body_height: float, 
#         body_mass: float
#         ) -> NDArray:

