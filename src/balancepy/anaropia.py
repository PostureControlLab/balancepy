# this submodule contains functions to analyze the data from anaropia balance experiments
import numpy as np
import balancepy.biomechanics as bm
import balancepy.timeseries as ts
import balancepy.data_class as data_class
from numpy.typing import NDArray as NDArray
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class AnaropiaPreprocessingConfig:
    """
    Configuration for Anaropia data preprocessing.
    
    Unified configuration class for both standard and legacy Anaropia data processing.
    All parameters have sensible defaults, allowing flexible use for different scenarios.
    
    Attributes
    ----------
    samplingrate_Hz : int, default=90
        Desired sampling rate in Hz. 0 means no resampling. Used for standard Anaropia.
    resample : bool, default=True
        If True, resample data to samplingrate_Hz.
    filter_type : str, default=None
        Type of filter to apply: 'low', 'high', 'band', or None for no filtering.
        Zeros phase filtering is applied to avoid phase shifts.
    filter_order : int, default=2
        Order of the filter.
    filter_cutoff_Hz : float, default=8
        Cutoff frequency for filtering. For 'band' filter_type, provide a tuple (low, high).
    cut_to_cycles : bool, default=True
        If True, cut data to cycles.
    end_time_seconds : float, default=260
        End time of the experiment in seconds for resampling.
        Standard: 260, Legacy: 220 (adjust as needed).
    cycle_start_samples : int, default=1800
        Start of the first cycle in samples (20*90 = 1800).
    cycle_length_samples : int, default=1800
        Cycle length in samples (20*90 = 1800).
    
    Examples
    --------
    Legacy Anaropia:
    
    >>> config = AnaropiaPreprocessingConfig(stimulus_name='stim_pitch', end_time_seconds=220)
    >>> com, time = getdata_legacy('legacy_data.csv', output='com', body_height_m=1.75, config=config)
    >>> stim, time = getdata_legacy('legacy_data.csv', output='stimulus', body_height_m=1.75, config=config)

    Standard Anaropia:
    
    >>> config = AnaropiaPreprocessingConfig()
    >>> com, time = getdata_anaropia('data.csv', output='com', direction='ap', body_height_m=1.75, config=config)
    >>> stim_ap, time = getdata_anaropia('data.csv', output='stimulus', direction='ap', body_height_m=1.75, config=config)
    >>> stim_ml, time = getdata_anaropia('data.csv', output='stimulus', direction='ml', body_height_m=1.75, config=config)
    """

    anaropia_version: str = 'legacy'  # 'legacy' or 'standard'
    samplingrate_Hz: int = 90
    resample: bool = True
    end_time_seconds: float = 260
    filter_type: str = None
    filter_order: int = 2
    filter_cutoff_Hz: float = 8
    cut_to_cycles: bool = False
    cycle_start_samples: int = 20*90
    cycle_length_samples: int = 20*90

    def __str__(self) -> str:
        """Pretty-print configuration settings."""
        settings = [
            f"samplingrate_Hz: {self.samplingrate_Hz}",
            f"resample: {self.resample}",
            f"end_time_seconds: {self.end_time_seconds}",
            f"filter_type: {self.filter_type}",
            f"filter_order: {self.filter_order}",
            f"filter_cutoff_Hz: {self.filter_cutoff_Hz}",
            f"cut_to_cycles: {self.cut_to_cycles}",
            f"cycle_start_samples: {self.cycle_start_samples}",
            f"cycle_length_samples: {self.cycle_length_samples}",
        ]
        return "AnaropiaPreprocessingConfig(\n  " + "\n  ".join(settings) + "\n)"


def getdata_anaropia(
    filename: str,
    output: str = 'com',
    direction: str = 'ap',
    body_height_m: float = None,
    body_weight_kg: float = None,
    config: AnaropiaPreprocessingConfig = None
) -> tuple:
    """
    Access and format data from balance experiments recorded with Anaropia.

    Reads data recorded using the Anaropia virtual-reality application for 
    balance experiments. Calculates stimulus and center of mass (COM) data.

    Parameters
    ----------
    filename : str
        Path and filename to be analyzed.
    output : str, default='com'
        Specifies which data column to return alongside time.
        - 'com'      : Center of mass sway (computed from shoulder/hip positions).
        - 'stimulus' : Stimulus signal (column determined by config.stimulus_name
                       and the direction parameter).
        - any other str : Raw column name from the data file (e.g. 'LeftShoulder_pos_z').
    body_height_m : float, optional
        Used for center of mass calculations.
    body_weight_kg : float, optional
        Used for center of pressure calculations.
    config : AnaropiaPreprocessingConfig, optional
        Configuration object containing processing parameters. If None, uses default
        configuration with standard settings.

    Returns
    -------
    signal : NDArray
        The requested data array (com, stimulus, or raw column).
    time : NDArray
        Time data.
    
    See Also
    --------
    AnaropiaPreprocessingConfig : Configuration class for Anaropia data preprocessing
    
    Examples
    --------
    >>> config = AnaropiaPreprocessingConfig()
    >>> com, time = getdata_anaropia('data.csv', output='com', direction='ap', body_height_m=1.75, config=config)
    >>> stim_ap, time = getdata_anaropia('data.csv', output='stimulus', direction='ap', body_height_m=1.75, config=config)
    >>> stim_ml, time = getdata_anaropia('data.csv', output='stimulus', direction='ml', body_height_m=1.75, config=config)
    """

    if config is None:
        config = AnaropiaPreprocessingConfig()

    # output_frequencies is a vector with the frequencies for which the FRF is calculated; default is up to 2 Hz
    # in case of the prts stimulus sequence, only every odd frequency point has energy, the even frequencies are zero

    raw_data = np.genfromtxt(filename, delimiter=',', names=True)

    if config.anaropia_version == 'standard':

        match output:
            case 'com':
                assert body_height_m is not None, "body_height_m must be provided for COM calculation"

                sho_height = np.mean(raw_data['LeftShoulder_pos_y'])
                hip_height = np.mean(raw_data['RightShoulder_pos_y'])

                if direction == 'ap':
                    sho = raw_data['LeftShoulder_pos_z']
                    hip = raw_data['RightShoulder_pos_z']
                elif direction == 'ml':
                    sho = raw_data['LeftShoulder_pos_x']
                    hip = raw_data['RightShoulder_pos_x']
                signal = bm.get_com(sho, sho_height, hip, hip_height, body_height_m, True)
            case 'Screen tilt' | 'stimulus':
                column_name = 'Screen_rot_x' if direction == 'ap' else 'Screen_rot_z'
                signal = raw_data[column_name]
            case 'data':
                signal = raw_data
            case _:
                assert output in raw_data.dtype.names, f"Output column '{output}' not found in data."
                signal = raw_data[output]

    elif config.anaropia_version == 'legacy':
        match output:
            case 'com':
                assert body_height_m is not None, "body_height_m must be provided for COM calculation"

                sho_height = np.mean(raw_data['shld_ypos'])
                hip_height = np.mean(raw_data['hip_ypos'])

                if direction == 'ap':
                    sho = raw_data['shld_zpos']
                    hip = raw_data['hip_zpos']
                elif direction == 'ml':
                    sho = raw_data['shld_xpos']
                    hip = raw_data['hip_xpos']
                signal = bm.get_com(sho, sho_height, hip, hip_height, body_height_m, True)

            case 'Screen tilt' | 'stimulus':
                column_name = 'stim_pitch' if direction == 'ap' else 'stim_roll'
                signal = raw_data[column_name]
            case 'data':
                signal = raw_data
            case _:
                assert output in raw_data.dtype.names, f"Output column '{output}' not found in data."
                signal = raw_data[output]


    # --- Extract time  ---
    time = raw_data['time']

    # --- Resample if requested ---
    if config.resample:
        signal, time = ts.resample(time, signal, config.samplingrate_Hz, config.end_time_seconds)

    # --- Apply filtering if requested ---
    if config.filter_type is not None:
        signal = ts.butterworth_filter(
            signal,
            samplingrate_Hz=config.samplingrate_Hz,
            filter_type=config.filter_type,
            order=config.filter_order,
            cutoff_Hz=config.filter_cutoff_Hz
        )

    # --- Cut to cycles if requested ---
    if config.cut_to_cycles:
        signal = ts.cut_to_cycles(signal, config.cycle_start_samples, config.cycle_length_samples)
        time = ts.cut_to_cycles(time, config.cycle_start_samples, config.cycle_length_samples)
    
    return signal, time




def plot_datacheck(
    filename: str,
    body_height_m: float,
    config: AnaropiaPreprocessingConfig = None,
    name: str = None,
    output_dir: Path | str = None,
    save: bool = True,
) -> tuple:
    """
    Plot data for comprehensive visual inspection of quality and structure.

    Generates a DinA4 landscape-formatted figure with multiple subplots for visually
    inspecting stimulus and response data using Plotly. The figure includes raw data
    translations, resampled stimulus and response with cycle indicators, and
    cycle-by-cycle analysis.

    Parameters
    ----------
    filename : str
        Path to the Anaropia data file (CSV format) to be plotted.
    body_height_m : float
        Subject body height in metres, used for COM calculation.
    config : AnaropiaPreprocessingConfig, optional
        Preprocessing configuration. If None, uses default settings.
    name : str, optional
        Label used as both the output filename stem (``datacheck_{name}.pdf``)
        and the figure title. If None, falls back to the input filename stem.
    output_dir : Path or str, optional
        Custom directory for saving the figure. If None, saves to the raw data file location.
    save : bool, default ``True``
        If ``True``, write the figure to the datacheck_plots directory.
        If ``False``, return the figure without saving.
        To skip existing files in a batch loop, check ``datacheck_output_path()``
        before calling this function.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    response_resampled : np.ndarray
    stimulus_resampled : np.ndarray

    See Also
    --------
    datacheck_output_path : Returns the output path without generating the figure.

    Examples
    --------
    >>> config = AnaropiaPreprocessingConfig()
    >>> fig, _, _ = plot_datacheck('data/raw/sMW09ON27_t5.csv', 1.75, config,
    ...                            name='MW09ON27_c5_s0_v1')
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from pathlib import Path
    from dataclasses import replace

    stem = name if name is not None else Path(filename).stem
    output_dir = Path(output_dir) if output_dir is not None else Path(filename).parent
    output_file = output_dir / f'datacheck_{stem}.png'

    if name is None:
        name = f'datacheck_{stem}.png'

    if config is None:
        config = AnaropiaPreprocessingConfig()

    # Create subplots with 4 rows: 3 full width + 1 with 4 subplots
    fig = make_subplots(
        rows=4, cols=4,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        column_widths=[0.25, 0.25, 0.25, 0.25],
        specs=[[{"colspan": 4}, None, None, None],
               [{"colspan": 4}, None, None, None],
               [{"colspan": 4}, None, None, None],
               [{}, {}, {}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # --- PLOT 1: Head, Shoulder, and Hip Translations (Raw Data) ---
    # Load raw data from file
    raw_data = np.genfromtxt(filename, delimiter=',', names=True)
    
    time_raw = raw_data['time']
    fig.add_trace(
        go.Scatter(x=time_raw, y=raw_data['zpos']-np.mean(raw_data['zpos']), name='head', mode='lines', 
                   line=dict(width=1.5), legend='legend'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_raw, y=raw_data['shld_zpos']-np.mean(raw_data['shld_zpos']), name='shoulder', mode='lines',
                   line=dict(width=1.5), legend='legend'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_raw, y=raw_data['hip_zpos']-np.mean(raw_data['hip_zpos']), name='hip', mode='lines',
                   line=dict(width=1.5), legend='legend'),
        row=1, col=1
    )
    fig.add_annotation(
        text=f"mean head: {np.mean(raw_data['zpos']):.4f}<br>mean sho: {np.mean(raw_data['shld_zpos']):.4f}<br>mean hip: {np.mean(raw_data['hip_zpos']):.4f}",
        xref="paper", yref="paper",
        x=0.99, y=1.05,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=9),
        align="left",
        xanchor="right",
        yanchor="top"
    )
    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_yaxes(title_text="ap translation (m)", row=1, col=1)
    
    # --- PLOT 2: Stimulus and Stimulus Resampled ---
    # Get not resampled data not cut to cycles
    config_not_resampled = replace(config, resample=False, cut_to_cycles=False)
    response_not_resampled, time_not_resampled = getdata_anaropia(filename, output='com', body_height_m=body_height_m, config=config_not_resampled)
    stimulus_not_resampled, _ = getdata_anaropia(filename, output='stimulus', body_height_m=body_height_m, config=config_not_resampled)

    # Get resampled data not cut to cycles
    config_resampled = replace(config, resample=True, cut_to_cycles=False)
    response_resampled, time_resampled = getdata_anaropia(filename, output='com', body_height_m=body_height_m, config=config_resampled)
    stimulus_resampled, _ = getdata_anaropia(filename, output='stimulus', body_height_m=body_height_m, config=config_resampled)

    fig.add_trace(
        go.Scatter(x=time_not_resampled, y=stimulus_not_resampled, name='recorded sampling', 
                   mode='lines', line=dict(width=0.5, color='blue'), 
                   opacity=0.7, legend='legend2'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_resampled, y=stimulus_resampled, name='resampled', 
                   mode='lines', line=dict(width=0.5, color='orange'), legend='legend2'),
        row=2, col=1
    )
    
    # Add cycle boundaries as vertical lines
    number_of_cycles = (config.samplingrate_Hz * config.end_time_seconds - config.cycle_start_samples) // config.cycle_length_samples
    for n in range(number_of_cycles):
        t = (config.cycle_start_samples + n * config.cycle_length_samples) / config.samplingrate_Hz
        fig.add_vline(x=t, line_dash="dash", line_color="red", line_width=0.5,
                        opacity=0.7, row=2, col=1)

    
    fig.update_xaxes(range=[0, config.end_time_seconds], row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(title_text="stimulus", row=2, col=1)
    
    # --- PLOT 3: Response (Center of Mass) and Response Resampled ---
    fig.add_trace(
        go.Scatter(x=time_not_resampled, y=response_not_resampled, name='raw response', 
                   mode='lines', line=dict(width=0.5, color='blue'), showlegend=False),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_resampled, y=response_resampled, name='resampled response', 
                   mode='lines', line=dict(width=0.5, color='orange'), showlegend=False),
        row=3, col=1
    )
    
    # Add cycle boundaries as vertical lines
    for n in range(number_of_cycles):
        t = (config.cycle_start_samples + n * config.cycle_length_samples) / config.samplingrate_Hz
        fig.add_vline(x=t, line_dash="dash", line_color="red", line_width=0.5, 
                        opacity=0.7, row=3, col=1)

    fig.update_xaxes(range=[0, config.end_time_seconds], row=3, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_yaxes(title_text="response (°)", row=3, col=1)
    
    # --- PLOT 4: Cycle-by-cycle analysis (4 subplots) ---
    # Get cycled data for sr_data object
    response_cycles, _ = getdata_anaropia(filename, output='com', body_height_m=body_height_m, config=config)
    stimulus_cycles, _ = getdata_anaropia(filename, output='stimulus', body_height_m=body_height_m, config=config)
    
    # Create sr_data object
    sr_data_obj = data_class.sr_data(
        samplingrate_Hz=config.samplingrate_Hz,
        stimulus=stimulus_cycles,
        response=response_cycles
    )

    # 4.1: Individual stimulus cycles and average
    if stimulus_cycles.ndim == 2:
        for i in range(stimulus_cycles.shape[1]):
            fig.add_trace(
                go.Scatter(x=sr_data_obj.time, y=stimulus_cycles[:, i], 
                          mode='lines', line=dict(width=0.5, color='grey'), 
                          opacity=0.6, showlegend=False),
                row=4, col=1
            )
        fig.add_trace(
            go.Scatter(x=sr_data_obj.time, y=sr_data_obj.stimulus_mean, name='Mean',
                      mode='lines', line=dict(width=2, color='darkblue'), showlegend=False),
            row=4, col=1
        )
    else:
        fig.add_trace(
            go.Scatter(x=sr_data_obj.time, y=sr_data_obj.stimulus_mean, mode='lines',
                      line=dict(width=2, color='darkblue'), showlegend=False),
            row=4, col=1
        )
    fig.update_xaxes(title_text="time (s)", row=4, col=1)
    fig.update_yaxes(title_text="stimulus (°)", row=4, col=1)
    
    # 4.2: Individual response cycles and average
    if response_cycles.ndim == 2:
        for i in range(response_cycles.shape[1]):
            fig.add_trace(
                go.Scatter(x=sr_data_obj.time, y=response_cycles[:, i],
                          mode='lines', line=dict(width=0.5, color='grey'),
                          opacity=0.6, showlegend=False),
                row=4, col=2
            )
        fig.add_trace(
            go.Scatter(x=sr_data_obj.time, y=sr_data_obj.response_mean, name='Mean',
                      mode='lines', line=dict(width=2, color='darkgreen'), showlegend=False),
            row=4, col=2
        )
    else:
        fig.add_trace(
            go.Scatter(x=sr_data_obj.time, y=sr_data_obj.response_mean, mode='lines',
                      line=dict(width=2, color='darkgreen'), showlegend=False),
            row=4, col=2
        )
    fig.update_xaxes(title_text="time (s)", row=4, col=2)
    fig.update_yaxes(title_text="response (°)", row=4, col=2)
    
    # 4.3: Stimulus amplitude spectra
    if sr_data_obj.stimulus_spectrum is not None:
        if sr_data_obj.stimulus_spectrum.ndim == 2:
            for i in range(sr_data_obj.stimulus_spectrum.shape[1]):
                fig.add_trace(
                    go.Scatter(x=sr_data_obj.freq, y=np.abs(sr_data_obj.stimulus_spectrum[:, i]),
                              mode='lines', line=dict(width=0.5, color='grey'),
                              opacity=0.6, showlegend=False),
                    row=4, col=3
                )
            fig.add_trace(
                go.Scatter(x=sr_data_obj.freq, y=np.abs(sr_data_obj.stimulus_spectrum_mean),
                          mode='lines', line=dict(width=2, color='darkblue'), name='Mean', showlegend=False),
                row=4, col=3
            )
        else:
            fig.add_trace(
                go.Scatter(x=sr_data_obj.freq, y=np.abs(sr_data_obj.stimulus_spectrum),
                          mode='lines', line=dict(width=2, color='darkblue'), showlegend=False),
                row=4, col=3
            )
    fig.update_xaxes(title_text="frequency (Hz)", row=4, col=3)
    fig.update_yaxes(title_text="amplitude (°)", row=4, col=3)
    
    # 4.4: Response amplitude spectra
    if sr_data_obj.response_spectrum is not None:
        if sr_data_obj.response_spectrum.ndim == 2:
            for i in range(sr_data_obj.response_spectrum.shape[1]):
                fig.add_trace(
                    go.Scatter(x=sr_data_obj.freq, y=np.abs(sr_data_obj.response_spectrum[:, i]),
                              mode='lines', line=dict(width=0.5, color='grey'),
                              opacity=0.6, showlegend=False),
                    row=4, col=4
                )
            fig.add_trace(
                go.Scatter(x=sr_data_obj.freq, y=np.abs(sr_data_obj.response_spectrum_mean),
                          mode='lines', line=dict(width=2, color='darkgreen'), name='Mean', showlegend=False),
                row=4, col=4
            )
        else:
            fig.add_trace(
                go.Scatter(x=sr_data_obj.freq, y=np.abs(sr_data_obj.response_spectrum),
                          mode='lines', line=dict(width=2, color='darkgreen'), showlegend=False),
                row=4, col=4
            )

    fig.update_xaxes(title_text="frequency (Hz)", row=4, col=4)
    fig.update_yaxes(title_text="amplitude (°)", row=4, col=4)
    
    # Update layout
    fig.update_layout(
        title_text=f'Data Check: {name}<br>File: {stem}',
        width=794, height=562,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h',
            x=0.8, y=0.78,
            xanchor='center', yanchor='middle',
            bgcolor='rgba(255,255,255,0.7)',
            font=dict(size=9),
        ),
        legend2=dict(
            orientation='h',
            x=0.8, y=0.50,
            xanchor='center', yanchor='middle',
            bgcolor='rgba(255,255,255,0.7)',
            font=dict(size=9),
        ),
        font=dict(size=10),
        title_font=dict(size=12),
        margin=dict(l=50, r=10, t=50, b=40),
    )
    fig.update_xaxes(tickfont=dict(size=9), title_font=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=9), title_font=dict(size=10))

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        # fig.write_image(str(output_file), width=794, height=562, scale=1)
        fig.write_image(str(output_file), width=794, height=562, scale=2)

    return fig, response_resampled, stimulus_resampled


def run_csmi(
    filename: str = None,
    body_height_m: float = None,
    body_weight_kg: float = None,
    name: str = None,
    config: AnaropiaPreprocessingConfig = None,
    *,
    com: np.ndarray = None,
    stimulus: np.ndarray = None
):
    """
    Run a CSMI (Continuous Sensory Manipulation Identification) analysis for one trial.

    Loads the stimulus and COM response from `filename`, constructs an sr_data object,
    fits the Peterka18 model, and returns the fitted model object.

    Parameters
    ----------
    filename : str
        Path to the Anaropia CSV data file.
    body_height_m : float
        Subject body height in metres.
    body_weight_kg : float
        Subject body weight in kilograms.
    config : AnaropiaPreprocessingConfig, optional
        Preprocessing configuration. Defaults to AnaropiaPreprocessingConfig().

    Returns
    -------
    subj : Peterka18
        Fitted Peterka18 model object with `.params`, `.fit_output`, and `.plot()`.

    Examples
    --------
    >>> config = bp.AnaropiaPreprocessingConfig()
    >>> config.end_time_seconds = 220
    >>> config.cut_to_cycles = True
    >>> subj = bp.run_csmi(filename, body_height_m=1.75, body_weight_kg=70, config=config)
    >>> subj.plot()
    """
    from balancepy.model_sim.peterka18 import Peterka18

    if config is None:
        config = AnaropiaPreprocessingConfig()

    if com is None or stimulus is None:
        _com, _ = getdata_anaropia(filename, output='com', body_height_m=body_height_m, config=config)
        _stim, _ = getdata_anaropia(filename, output='stimulus', config=config)
    else:
        if config.cut_to_cycles:
            _com = ts.cut_to_cycles(com, config.cycle_start_samples, config.cycle_length_samples)
            _stim = ts.cut_to_cycles(stimulus, config.cycle_start_samples, config.cycle_length_samples)
        else:
            _com, _stim = com, stimulus

    if name is None and filename is not None:
        from pathlib import Path as _Path
        _parts = _Path(filename).parts
        name = str(_Path(*_parts[-4:])) if len(_parts) >= 4 else filename

    data_exp = data_class.sr_data(
        samplingrate_Hz=config.samplingrate_Hz,
        stimulus=_stim,
        response=_com,
        frequency_selection='prts',
        name=name
    )

    subj = Peterka18(body_weight_kg, body_height_m, data_exp=data_exp)
    subj.fit()
    return subj


