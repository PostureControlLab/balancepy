import numpy as np
import balancepy as bp
import balancepy.biomechanics as bm
import balancepy.timeseries as ts
import balancepy.frequency as fd
from balancepy.frequency import frequency_response_function as get_frf
from numpy.typing import NDArray as NDArray
import balancepy.models.Peterka2018 as Peterka2018
import numpy.lib.recfunctions as rfn
import balancepy.models.ICdual as ICdual

def prts_analysis(
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
    # output_frequencies is a vector with the frequencies for which the FRF is calculated; default is up to 2 Hz
    # in case of the prts stimulus sequence, only every odd frequency point has energy, the even frequencies are zero
    
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

    # run model simulations
    opts = Peterka2018.getOpts_ICmodel_Peterka2018(body_mass, body_height)
    par_out, sim_frf, res = Peterka2018.fit_ICmodel_maxLikelihood(FD, opts)
    FD = np.column_stack((FD, sim_frf))

    return FD, TD, par_out


def lifespan_analysis(
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
        NDArray: 2D experimental frequency domain data longer stimulus component
        NDArray: 2D experimental time domain data longer stimulus component
        NDArray: 2D experimental frequency domain data shorter stimulus component
        NDArray: 2D experimental time domain data shorter stimulus component
    """

    sampling_rate = 90 # numer gives desired sampling rate; 0 means no resampling
    end_time = 220
    cycle_start_samples = 20*sampling_rate
    cycle_length_samples = 20*sampling_rate
    # output_frequencies is a vector with the frequencies for which the FRF is calculated; default is up to 2 Hz
    # in case of the prts stimulus sequence, only every odd frequency point has energy, the even frequencies are zero
    
    data = np.genfromtxt(fname, delimiter=',', names=True)

    time_raw = data['time']
    stim_vis_raw = data['stim_pitch']
    stim_surf_raw = data['analog4']
    
    com_raw = bm.com(data['shld_zpos'], np.mean(data['shld_ypos']), data['hip_zpos'], np.mean(data['hip_ypos']),body_height,True)

    com = ts.resample(time_raw, com_raw, sampling_rate, end_time)
    stim_vis = ts.resample(time_raw, stim_vis_raw, sampling_rate, end_time)
    stim_prop = ts.resample(time_raw, stim_surf_raw, sampling_rate, end_time)

    com_cyc = ts.cut_to_cycles(com, cycle_start_samples, cycle_length_samples)
    stim_vis_cyc = ts.cut_to_cycles(stim_vis, cycle_start_samples, cycle_length_samples)
    stim_prop_cyc = ts.cut_to_cycles(stim_prop, cycle_start_samples, cycle_length_samples)

    selected_frequencies = range(0, int(2 * np.size(com_cyc, 0) / sampling_rate), 2)
    FDprop, TDprop = get_frf(stim_prop_cyc, com_cyc, sampling_rate, selected_frequencies)

    selected_frequencies = range(1, int(2 * np.size(com_cyc, 0) / sampling_rate), 4)
    FDvis, TDvis = get_frf(stim_vis_cyc, com_cyc, sampling_rate, selected_frequencies)

    # run model simulations
    opts = ICdual.settings(body_mass, body_height)
    par_out, sim_prop, sim_vis, res = ICdual.fit(FDprop, FDvis, opts)

    FDprop = rfn.append_fields(FDprop, 'sim_frf', sim_prop, usemask=False)
    FDvis = rfn.append_fields(FDvis, 'sim_frf', sim_vis, usemask=False)


    return FDprop, TDprop, FDvis, TDvis, par_out