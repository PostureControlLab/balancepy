# this submodule contains functions to analyze the data from balance experiments
import warnings
import numpy as np
import balancepy.biomechanics as bm
import balancepy.timeseries as ts
import balancepy.data_class as data_class
from numpy.typing import NDArray as NDArray
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, replace, field


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnaropiaPreprocessingConfig:
    """
    Configuration for a generic 1D signal preprocessing pipeline.

    Controls resampling, filtering, and cycle-cutting applied to any 1D time
    series via :func:`balancepy.timeseries.preprocess`.

    Attributes
    ----------
    samplingrate_Hz : int
        Target sampling rate in Hz.
    resample : bool
        If True, resample the signal to *samplingrate_Hz*.
    end_time_seconds : float
        End time for resampling in seconds.
    filter_type : str or None
        Butterworth filter type: ``'lowpass'``, ``'highpass'``, ``'bandpass'``,
        ``'bandstop'``, or ``None`` for no filtering.
    filter_order : int
        Butterworth filter order.
    filter_cutoff_Hz : float or list[float]
        Cutoff frequency (scalar for low/highpass, 2-element list for band).
    cut_to_cycles : bool
        If True, segment the signal into cycles.
    cycle_start_samples : int
        First sample of the first cycle.
    cycle_length_samples : int
        Length of each cycle in samples.
    """

    samplingrate_Hz: int = 90
    resample: bool = True
    end_time_seconds: float = 260
    filter_type: str = None
    filter_order: int = 2
    filter_cutoff_Hz: float = 8
    cut_to_cycles: bool = False
    cycle_start_samples: int = 20 * 90
    cycle_length_samples: int = 20 * 90

    def __post_init__(self):
        _valid_filters = {None, 'lowpass', 'highpass', 'bandpass', 'bandstop'}
        if self.filter_type not in _valid_filters:
            raise ValueError(
                f"filter_type must be one of {_valid_filters}, got {self.filter_type!r}"
            )

    def __str__(self) -> str:
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


@dataclass
class COMConfig:
    """
    Column-name layout for centre-of-mass (COM) computation.

    Describes which CSV columns contain the marker positions needed by
    :func:`balancepy.biomechanics.get_com`.  Subject-specific values
    (``body_height_m``) are **not** stored here — they are function parameters.

    Attributes
    ----------
    shoulder_pos_column : str
        Column name for the shoulder position in the sway direction.
    hip_pos_column : str
        Column name for the hip position in the sway direction.
    shoulder_height_column : str
        Column name for the shoulder marker height (vertical axis).
    hip_height_column : str
        Column name for the hip marker height (vertical axis).
    rotation : bool
        If True, :func:`~balancepy.biomechanics.get_com` returns COM angle
        in degrees; if False, linear displacement in metres.
    """

    shoulder_pos_column: str
    hip_pos_column: str
    shoulder_height_column: str
    hip_height_column: str
    rotation: bool = True

# Predefined COMConfig instances
COM_LEGACY_AP = COMConfig('shld_zpos', 'hip_zpos', 'shld_ypos', 'hip_ypos')
COM_LEGACY_ML = COMConfig('shld_xpos', 'hip_xpos', 'shld_ypos', 'hip_ypos')
COM_STANDARD_AP = COMConfig('LeftShoulder_pos_z', 'RightShoulder_pos_z',
                            'LeftShoulder_pos_y', 'RightShoulder_pos_y')
COM_STANDARD_ML = COMConfig('LeftShoulder_pos_x', 'RightShoulder_pos_x',
                            'LeftShoulder_pos_y', 'RightShoulder_pos_y')


@dataclass
class AnaropiaSRDataConfig:
    """
    Defines how stimulus and response signals are obtained from an Anaropia CSV
    and how the resulting :class:`~balancepy.data_class.sr_data` object is
    configured.

    Exactly one of *response_column* or *com_config* must be provided.
    When *com_config* is set the response is computed as the centre of mass
    via :func:`~balancepy.biomechanics.get_com`.

    Attributes
    ----------
    stimulus_column : str or tuple of (str, str)
        CSV column name for the stimulus signal. When a 2-tuple is given,
        the first element is the primary stimulus used for analysis and the
        second is an additional channel displayed in :func:`plot_datacheck`.
    response_column : str or None
        CSV column name for a direct response signal.
    com_config : COMConfig or None
        If the response is COM, provide marker-column layout here.
    frequency_selection : str
        Frequency selection mode passed to :class:`~balancepy.data_class.sr_data`.
    name : str or None
        Optional label forwarded to ``sr_data.name``.
    column_scales : dict mapping str → float, optional
        Per-column scale factors applied when reading CSV columns.
        For example ``{'analog4': -1.0}`` inverts that channel.
    """

    stimulus_column: "str | tuple[str, str]"
    response_column: str | None = None
    com_config: COMConfig | None = None
    frequency_selection: str = 'prts'
    name: str | None = None
    column_scales: dict[str, float] | None = None

    def __post_init__(self):
        has_col = self.response_column is not None
        has_com = self.com_config is not None
        if has_col == has_com:
            raise ValueError(
                "Exactly one of 'response_column' or 'com_config' must be set, "
                f"got response_column={self.response_column!r}, "
                f"com_config={'set' if has_com else None}."
            )

# Predefined AnaropiaSRDataConfig instances
SR_LEGACY_AP = AnaropiaSRDataConfig('stim_pitch', com_config=COM_LEGACY_AP)
SR_LEGACY_ML = AnaropiaSRDataConfig('stim_roll', com_config=COM_LEGACY_ML)
SR_STANDARD_AP = AnaropiaSRDataConfig('Screen_rot_x', com_config=COM_STANDARD_AP)
SR_STANDARD_ML = AnaropiaSRDataConfig('Screen_rot_z', com_config=COM_STANDARD_ML)


# ---------------------------------------------------------------------------
# Backward-compatible alias (deprecated)
# ---------------------------------------------------------------------------
# class AnaropiaPreprocessingConfig(PreprocessingConfig):
#     """Deprecated — use :class:`PreprocessingConfig` instead."""

#     def __init__(self, anaropia_version: str = 'legacy', **kwargs):
#         warnings.warn(
#             "AnaropiaPreprocessingConfig is deprecated. "
#             "Use PreprocessingConfig + AnaropiaSRDataConfig instead.",
#             DeprecationWarning,
#             stacklevel=2,
#         )
#         super().__init__(**kwargs)
#         self.anaropia_version = anaropia_version
# 
# def getdata_anaropia(
#     filename: str,
#     output: str = 'com',
#     direction: str = 'ap',
#     body_height_m: float = None,
#     body_weight_kg: float = None,
#     config: AnaropiaPreprocessingConfig = None
# ) -> tuple:
#     """
#     Access and format data from balance experiments recorded with Anaropia.

#     Reads data recorded using the Anaropia virtual-reality application for 
#     balance experiments. Calculates stimulus and center of mass (COM) data.

#     Parameters
#     ----------
#     filename : str
#         Path and filename to be analyzed.
#     output : str, default='com'
#         Specifies which data column to return alongside time.
#         - 'com'      : Center of mass sway (computed from shoulder/hip positions).
#         - 'stimulus' : Stimulus signal (column determined by config.stimulus_name
#                        and the direction parameter).
#         - any other str : Raw column name from the data file (e.g. 'LeftShoulder_pos_z').
#     body_height_m : float, optional
#         Used for center of mass calculations.
#     body_weight_kg : float, optional
#         Used for center of pressure calculations.
#     config : AnaropiaPreprocessingConfig, optional
#         Configuration object containing processing parameters. If None, uses default
#         configuration with standard settings.

#     Returns
#     -------
#     signal : NDArray
#         The requested data array (com, stimulus, or raw column).
#     time : NDArray
#         Time data.
    
#     See Also
#     --------
#     AnaropiaPreprocessingConfig : Configuration class for Anaropia data preprocessing
    
#     Examples
#     --------
#     >>> config = AnaropiaPreprocessingConfig()
#     >>> com, time = getdata_anaropia('data.csv', output='com', direction='ap', body_height_m=1.75, config=config)
#     >>> stim_ap, time = getdata_anaropia('data.csv', output='stimulus', direction='ap', body_height_m=1.75, config=config)
#     >>> stim_ml, time = getdata_anaropia('data.csv', output='stimulus', direction='ml', body_height_m=1.75, config=config)
#     """

#     warnings.warn(
#         "getdata_anaropia() is deprecated. Use _extract_stimulus / "
#         "_extract_response + ts.preprocess() instead.",
#         DeprecationWarning,
#         stacklevel=2,
#     )

#     if config is None:
#         config = AnaropiaPreprocessingConfig()

#     # output_frequencies is a vector with the frequencies for which the FRF is calculated; default is up to 2 Hz
#     # in case of the prts stimulus sequence, only every odd frequency point has energy, the even frequencies are zero

#     raw_data = np.genfromtxt(filename, delimiter=',', names=True)

#     if config.anaropia_version == 'standard':

#         match output:
#             case 'com':
#                 assert body_height_m is not None, "body_height_m must be provided for COM calculation"

#                 sho_height = np.mean(raw_data['LeftShoulder_pos_y'])
#                 hip_height = np.mean(raw_data['RightShoulder_pos_y'])

#                 if direction == 'ap':
#                     sho = raw_data['LeftShoulder_pos_z']
#                     hip = raw_data['RightShoulder_pos_z']
#                 elif direction == 'ml':
#                     sho = raw_data['LeftShoulder_pos_x']
#                     hip = raw_data['RightShoulder_pos_x']
#                 signal = bm.get_com(sho, sho_height, hip, hip_height, body_height_m, True)
#             case 'Screen tilt' | 'stimulus':
#                 column_name = 'Screen_rot_x' if direction == 'ap' else 'Screen_rot_z'
#                 signal = raw_data[column_name]
#             case 'data':
#                 signal = raw_data
#             case _:
#                 assert output in raw_data.dtype.names, f"Output column '{output}' not found in data."
#                 signal = raw_data[output]

#     elif config.anaropia_version == 'legacy':
#         match output:
#             case 'com':
#                 assert body_height_m is not None, "body_height_m must be provided for COM calculation"

#                 sho_height = np.mean(raw_data['shld_ypos'])
#                 hip_height = np.mean(raw_data['hip_ypos'])

#                 if direction == 'ap':
#                     sho = raw_data['shld_zpos']
#                     hip = raw_data['hip_zpos']
#                 elif direction == 'ml':
#                     sho = raw_data['shld_xpos']
#                     hip = raw_data['hip_xpos']
#                 signal = bm.get_com(sho, sho_height, hip, hip_height, body_height_m, True)

#             case 'Screen tilt' | 'stimulus':
#                 column_name = 'stim_pitch' if direction == 'ap' else 'stim_roll'
#                 signal = raw_data[column_name]
#             case 'data':
#                 signal = raw_data
#             case _:
#                 assert output in raw_data.dtype.names, f"Output column '{output}' not found in data."
#                 signal = raw_data[output]


#     # --- Extract time  ---
#     time = raw_data['time']

#     # --- Resample if requested ---
#     if config.resample:
#         signal, time = ts.resample(time, signal, config.samplingrate_Hz, config.end_time_seconds)

#     # --- Apply filtering if requested ---
#     if config.filter_type is not None:
#         signal = ts.butterworth_filter(
#             signal,
#             samplingrate_Hz=config.samplingrate_Hz,
#             filter_type=config.filter_type,
#             order=config.filter_order,
#             cutoff_Hz=config.filter_cutoff_Hz
#         )

#     # --- Cut to cycles if requested ---
#     if config.cut_to_cycles:
#         signal = ts.cut_to_cycles(signal, config.cycle_start_samples, config.cycle_length_samples)
#         time = ts.cut_to_cycles(time, config.cycle_start_samples, config.cycle_length_samples)
    
#     return signal, time

def preprocess(signal, time, config):
    """
    Generic 1D preprocessing pipeline: resample → filter → cut to cycles.

    Parameters
    ----------
    signal : NDArray
        1D input signal.
    time : NDArray
        1D time vector corresponding to *signal*.
    config : AnaropiaPreprocessingConfig
        Pipeline settings (from :mod:`balancepy.anaropia`).

    Returns
    -------
    signal : NDArray
        Processed signal (1D or 2D if cut to cycles).
    time : NDArray
        Corresponding time vector (1D or 2D if cut to cycles).
    """
    if config.resample:
        signal, time = ts.resample(time, signal, config.samplingrate_Hz,
                                   config.end_time_seconds)

    if config.filter_type is not None:
        signal = ts.butterworth_filter(
            signal,
            samplingrate_Hz=config.samplingrate_Hz,
            filter_type=config.filter_type,
            order=config.filter_order,
            cutoff_Hz=config.filter_cutoff_Hz,
        )

    if config.cut_to_cycles:
        signal = ts.cut_to_cycles(signal, config.cycle_start_samples,
                                  config.cycle_length_samples)
        time = ts.cut_to_cycles(time, config.cycle_start_samples,
                                config.cycle_length_samples)

    return signal, time


# ---------------------------------------------------------------------------
# Internal helpers for extracting signals from raw CSV data
# ---------------------------------------------------------------------------

def _get_column(raw_data, col_name: str, sr_config: AnaropiaSRDataConfig):
    """Read a column from *raw_data*, applying any scale from *sr_config.column_scales*."""
    data = raw_data[col_name].copy()
    if sr_config.column_scales and col_name in sr_config.column_scales:
        data = data * sr_config.column_scales[col_name]
    return data


def _extract_stimulus(raw_data, sr_config: AnaropiaSRDataConfig):
    """Return the 1D stimulus array from *raw_data* using *sr_config*."""
    col = sr_config.stimulus_column
    if isinstance(col, tuple):
        col = col[0]
    return _get_column(raw_data, col, sr_config)


def _extract_response(raw_data, sr_config: AnaropiaSRDataConfig, body_height_m: float = None):
    """Return the 1D response array from *raw_data*.

    If *sr_config.com_config* is set, computes COM via :func:`bm.get_com`;
    otherwise reads the column named by *sr_config.response_column*.
    """
    if sr_config.com_config is not None:
        cc = sr_config.com_config
        if body_height_m is None:
            raise ValueError(
                "body_height_m is required when sr_config uses a COMConfig response."
            )
        sho = _get_column(raw_data, cc.shoulder_pos_column, sr_config)
        hip = _get_column(raw_data, cc.hip_pos_column, sr_config)
        sho_height = np.mean(_get_column(raw_data, cc.shoulder_height_column, sr_config))
        hip_height = np.mean(_get_column(raw_data, cc.hip_height_column, sr_config))
        return bm.get_com(sho, sho_height, hip, hip_height, body_height_m, cc.rotation)
    else:
        return _get_column(raw_data, sr_config.response_column, sr_config)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def plot_datacheck(
    filename: str,
    body_height_m: float,
    sr_config: AnaropiaSRDataConfig = None,
    preproc_config: AnaropiaPreprocessingConfig = None,
    *,
    name: str = None,
    output_dir: Path | str = None,
    save: bool = True,
    # Deprecated keyword — accepted but ignored when sr_config is provided
    config: "AnaropiaPreprocessingConfig | None" = None,
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
    sr_config : AnaropiaSRDataConfig, optional
        Stimulus/response configuration. Defaults to ``SR_LEGACY_AP``.
    preproc_config : AnaropiaPreprocessingConfig, optional
        Preprocessing configuration. Defaults to ``AnaropiaPreprocessingConfig()``.
    name : str, optional
        Label used as both the output filename stem (``datacheck_{name}.png``)
        and the figure title. If None, falls back to the input filename stem.
    output_dir : Path or str, optional
        Custom directory for saving the figure. If None, saves to the raw data
        file location.
    save : bool, default True
        If True, write the figure to disk as PNG.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    response_resampled : np.ndarray
    stimulus_resampled : np.ndarray

    Examples
    --------
    >>> fig, _, _ = plot_datacheck(
    ...     'data/raw/sMW09ON27_t5.csv', 1.75,
    ...     sr_config=bp.SR_LEGACY_AP,
    ...     preproc_config=bp.AnaropiaPreprocessingConfig(end_time_seconds=220),
    ...     name='MW09ON27_c5_s0_v1',
    ... )
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    if sr_config is None:
        sr_config = SR_LEGACY_AP
    if preproc_config is None:
        preproc_config = AnaropiaPreprocessingConfig()

    stem = name if name is not None else Path(filename).stem
    _output_dir = Path(output_dir) if output_dir is not None else Path(filename).parent
    output_file = _output_dir / f'datacheck_{stem}.png'

    if name is None:
        name = f'datacheck_{stem}.png'

    # ------------------------------------------------------------------
    # Load raw data once; extract signals once
    # ------------------------------------------------------------------
    raw_data = np.genfromtxt(filename, delimiter=',', names=True)
    time_raw = raw_data['time']

    stimulus_raw = _extract_stimulus(raw_data, sr_config)
    response_raw = _extract_response(raw_data, sr_config, body_height_m)

    # Preprocess at two stages:
    #   1) resampled (+ filtered) but NOT cut to cycles
    #   2) resampled (+ filtered) AND cut to cycles
    config_resampled = replace(preproc_config, cut_to_cycles=False)
    stimulus_resampled, time_resampled = preprocess(stimulus_raw, time_raw, config_resampled)
    response_resampled, _ = preprocess(response_raw, time_raw, config_resampled)

    # For cycles, only cut (do not re-resample/re-filter the already-processed data)
    config_cycles_only = AnaropiaPreprocessingConfig(
        resample=False,
        filter_type=None,
        cut_to_cycles=True,
        cycle_start_samples=preproc_config.cycle_start_samples,
        cycle_length_samples=preproc_config.cycle_length_samples,
    )
    stimulus_cycles, _ = preprocess(stimulus_resampled, time_resampled, config_cycles_only)
    response_cycles, _ = preprocess(response_resampled, time_resampled, config_cycles_only)

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=4, cols=4,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        column_widths=[0.25, 0.25, 0.25, 0.25],
        specs=[[{"colspan": 4}, None, None, None],
               [{"colspan": 4}, None, None, None],
               [{"colspan": 4}, None, None, None],
               [{}, {}, {}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # --- PLOT 1: Raw marker trajectories --------------------------------
    # Use COMConfig columns for shoulder/hip; add head marker if present
    cc = sr_config.com_config
    if cc is not None:
        sho_col = cc.shoulder_pos_column
        hip_col = cc.hip_pos_column
        # Hardcoded head-marker column (legacy: 'zpos')
        head_col = 'zpos' if sho_col.startswith('shld_') else None

        if head_col is not None and head_col in raw_data.dtype.names:
            fig.add_trace(
                go.Scatter(x=time_raw,
                           y=raw_data[head_col] - np.mean(raw_data[head_col]),
                           name='head', mode='lines', line=dict(width=1.5),
                           legend='legend'),
                row=1, col=1,
            )
        fig.add_trace(
            go.Scatter(x=time_raw,
                       y=raw_data[sho_col] - np.mean(raw_data[sho_col]),
                       name='shoulder', mode='lines', line=dict(width=1.5),
                       legend='legend'),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=time_raw,
                       y=raw_data[hip_col] - np.mean(raw_data[hip_col]),
                       name='hip', mode='lines', line=dict(width=1.5),
                       legend='legend'),
            row=1, col=1,
        )

        annotation_parts = []
        if head_col is not None and head_col in raw_data.dtype.names:
            annotation_parts.append(f"mean head: {np.mean(raw_data[head_col]):.4f}")
        annotation_parts.append(f"mean sho: {np.mean(raw_data[sho_col]):.4f}")
        annotation_parts.append(f"mean hip: {np.mean(raw_data[hip_col]):.4f}")
        fig.add_annotation(
            text="<br>".join(annotation_parts),
            xref="paper", yref="paper", x=0.99, y=1.05,
            showarrow=False, bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black", borderwidth=1, font=dict(size=9),
            align="left", xanchor="right", yanchor="top",
        )
    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_yaxes(title_text="ap translation (m)", row=1, col=1)

    # --- PLOT 2: Stimulus (original + resampled) -------------------------
    fig.add_trace(
        go.Scatter(x=time_raw, y=stimulus_raw, name='recorded sampling',
                   mode='lines', line=dict(width=0.5, color='blue'),
                   opacity=0.7, legend='legend2'),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_resampled, y=stimulus_resampled, name='resampled',
                   mode='lines', line=dict(width=0.5, color='orange'),
                   legend='legend2'),
        row=2, col=1,
    )

    # Overlay second stimulus channel when stimulus_column is a 2-tuple
    if isinstance(sr_config.stimulus_column, tuple) and len(sr_config.stimulus_column) > 1:
        _extra_col = sr_config.stimulus_column[1]
        fig.add_trace(
            go.Scatter(x=time_raw, y=_get_column(raw_data, _extra_col, sr_config),
                       name=_extra_col, mode='lines',
                       line=dict(width=0.5, color='green'),
                       opacity=0.7, legend='legend2'),
            row=2, col=1,
        )

    number_of_cycles = int(
        (preproc_config.samplingrate_Hz * preproc_config.end_time_seconds
         - preproc_config.cycle_start_samples) // preproc_config.cycle_length_samples
    )
    for n in range(number_of_cycles):
        t = (preproc_config.cycle_start_samples
             + n * preproc_config.cycle_length_samples) / preproc_config.samplingrate_Hz
        fig.add_vline(x=t, line_dash="dash", line_color="red", line_width=0.5,
                      opacity=0.7, row=2, col=1)

    fig.update_xaxes(range=[0, preproc_config.end_time_seconds], row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_yaxes(title_text="stimulus", row=2, col=1)

    # --- PLOT 3: Response (original + resampled) -------------------------
    fig.add_trace(
        go.Scatter(x=time_raw, y=response_raw, name='raw response',
                   mode='lines', line=dict(width=0.5, color='blue'),
                   showlegend=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_resampled, y=response_resampled,
                   name='resampled response', mode='lines',
                   line=dict(width=0.5, color='orange'), showlegend=False),
        row=3, col=1,
    )

    for n in range(number_of_cycles):
        t = (preproc_config.cycle_start_samples
             + n * preproc_config.cycle_length_samples) / preproc_config.samplingrate_Hz
        fig.add_vline(x=t, line_dash="dash", line_color="red", line_width=0.5,
                      opacity=0.7, row=3, col=1)

    fig.update_xaxes(range=[0, preproc_config.end_time_seconds], row=3, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_yaxes(title_text="response (°)", row=3, col=1)

    # --- PLOT 4: Cycle-by-cycle analysis (4 subplots) -------------------
    # Build one sr_data per stimulus column
    stim_cols = sr_config.stimulus_column
    if isinstance(stim_cols, str):
        stim_cols = (stim_cols,)

    _stim_colors = ['darkblue', 'darkred', 'darkorange', 'purple']
    _stim_colors_light = ['lightblue', 'lightsalmon', 'moccasin', 'plum']
    multi = len(stim_cols) > 1

    sr_data_objects: list[tuple[str, "data_class.sr_data"]] = []
    for col in stim_cols:
        stim_raw_col = _get_column(raw_data, col, sr_config)
        stim_resampled_col, _ = preprocess(stim_raw_col, time_raw, config_resampled)
        stim_cycles_col, _ = preprocess(stim_resampled_col, time_resampled, config_cycles_only)
        sr_obj = data_class.sr_data(
            samplingrate_Hz=preproc_config.samplingrate_Hz,
            stimulus=stim_cycles_col,
            response=response_cycles,
        )
        sr_data_objects.append((col, sr_obj))

    # 4.1: Individual stimulus cycles and average (all stimulus channels)
    for idx, (col_name, sr_obj) in enumerate(sr_data_objects):
        color = _stim_colors[idx % len(_stim_colors)]
        color_light = _stim_colors_light[idx % len(_stim_colors_light)]
        stim = sr_obj.stimulus
        if stim.ndim == 2:
            for i in range(stim.shape[1]):
                fig.add_trace(
                    go.Scatter(x=sr_obj.time, y=stim[:, i],
                               mode='lines', line=dict(width=0.5, color=color_light),
                               opacity=0.6, showlegend=False),
                    row=4, col=1,
                )
        fig.add_trace(
            go.Scatter(x=sr_obj.time, y=sr_obj.stimulus_mean,
                       name=col_name if multi else 'Mean', mode='lines',
                       line=dict(width=2, color=color),
                       showlegend=multi,
                       **(dict(legend='legend3') if multi else {})),
            row=4, col=1,
        )
    fig.update_xaxes(title_text="time (s)", row=4, col=1)
    fig.update_yaxes(title_text="stimulus (°)", row=4, col=1)

    # 4.2: Individual response cycles and average (shared across stimuli)
    sr_data_primary = sr_data_objects[0][1]
    if response_cycles.ndim == 2:
        for i in range(response_cycles.shape[1]):
            fig.add_trace(
                go.Scatter(x=sr_data_primary.time, y=response_cycles[:, i],
                           mode='lines', line=dict(width=0.5, color='grey'),
                           opacity=0.6, showlegend=False),
                row=4, col=2,
            )
        fig.add_trace(
            go.Scatter(x=sr_data_primary.time, y=sr_data_primary.response_mean,
                       name='Mean', mode='lines',
                       line=dict(width=2, color='darkgreen'), showlegend=False),
            row=4, col=2,
        )
    else:
        fig.add_trace(
            go.Scatter(x=sr_data_primary.time, y=sr_data_primary.response_mean,
                       mode='lines', line=dict(width=2, color='darkgreen'),
                       showlegend=False),
            row=4, col=2,
        )
    fig.update_xaxes(title_text="time (s)", row=4, col=2)
    fig.update_yaxes(title_text="response (°)", row=4, col=2)

    # 4.3: Stimulus amplitude spectra — stem plot (all stimulus channels)
    for idx, (col_name, sr_obj) in enumerate(sr_data_objects):
        color = _stim_colors[idx % len(_stim_colors)]
        if sr_obj.stimulus_spectrum is not None:
            if sr_obj.stimulus_spectrum.ndim == 2:
                for i in range(sr_obj.stimulus_spectrum.shape[1]):
                    amp = np.abs(sr_obj.stimulus_spectrum[:, i])
                    fig.add_trace(
                        go.Scatter(x=sr_obj.freq, y=amp,
                                   mode='markers',
                                   marker=dict(size=3, color='grey'),
                                   opacity=0.5, showlegend=False),
                        row=4, col=3,
                    )
                amp_mean = np.abs(sr_obj.stimulus_spectrum_mean)
            else:
                amp_mean = np.abs(sr_obj.stimulus_spectrum)

            # Stem lines (None-separated segments from 0 to amplitude)
            xs = np.repeat(sr_obj.freq, 3)
            ys = np.empty(len(sr_obj.freq) * 3)
            ys[0::3] = 0
            ys[1::3] = amp_mean
            ys[2::3] = np.nan
            fig.add_trace(
                go.Scatter(x=xs, y=ys, mode='lines',
                           line=dict(width=1.5, color=color),
                           showlegend=False),
                row=4, col=3,
            )
            # Marker caps
            fig.add_trace(
                go.Scatter(x=sr_obj.freq, y=amp_mean,
                           mode='markers',
                           marker=dict(size=5, color=color),
                           name=col_name if multi else 'Mean',
                           showlegend=False),
                row=4, col=3,
            )
    fig.update_xaxes(title_text="frequency (Hz)", row=4, col=3)
    fig.update_yaxes(title_text="amplitude (°)", row=4, col=3)

    # 4.4: Response amplitude spectra — stem plot (shared across stimuli)
    if sr_data_primary.response_spectrum is not None:
        if sr_data_primary.response_spectrum.ndim == 2:
            for i in range(sr_data_primary.response_spectrum.shape[1]):
                amp = np.abs(sr_data_primary.response_spectrum[:, i])
                fig.add_trace(
                    go.Scatter(x=sr_data_primary.freq, y=amp,
                               mode='markers',
                               marker=dict(size=3, color='grey'),
                               opacity=0.5, showlegend=False),
                    row=4, col=4,
                )
            amp_mean = np.abs(sr_data_primary.response_spectrum_mean)
        else:
            amp_mean = np.abs(sr_data_primary.response_spectrum)

        # Stem lines
        xs = np.repeat(sr_data_primary.freq, 3)
        ys = np.empty(len(sr_data_primary.freq) * 3)
        ys[0::3] = 0
        ys[1::3] = amp_mean
        ys[2::3] = np.nan
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode='lines',
                       line=dict(width=1.5, color='darkgreen'),
                       showlegend=False),
            row=4, col=4,
        )
        # Marker caps
        fig.add_trace(
            go.Scatter(x=sr_data_primary.freq, y=amp_mean,
                       mode='markers',
                       marker=dict(size=5, color='darkgreen'),
                       name='Mean', showlegend=False),
            row=4, col=4,
        )

    fig.update_xaxes(title_text="frequency (Hz)", row=4, col=4)
    fig.update_yaxes(title_text="amplitude (°)", row=4, col=4)

    # --- Layout -----------------------------------------------------------
    fig.update_layout(
        title_text=f'Data Check: {name}<br>File: {stem}',
        width=794, height=562,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation='h', x=0.8, y=0.78,
            xanchor='center', yanchor='middle',
            bgcolor='rgba(255,255,255,0.7)', font=dict(size=9),
        ),
        legend2=dict(
            orientation='h', x=0.8, y=0.50,
            xanchor='center', yanchor='middle',
            bgcolor='rgba(255,255,255,0.7)', font=dict(size=9),
        ),
        legend3=dict(
            orientation='h', x=0.8, y=0.22,
            xanchor='center', yanchor='middle',
            bgcolor='rgba(255,255,255,0.7)', font=dict(size=9),
        ),
        font=dict(size=10),
        title_font=dict(size=12),
        margin=dict(l=50, r=10, t=50, b=40),
    )
    fig.update_xaxes(tickfont=dict(size=9), title_font=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=9), title_font=dict(size=10))

    if save:
        _output_dir.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_file), width=794, height=562, scale=2)

    return fig, response_resampled, stimulus_resampled

def run_csmi(
    filename: str = None,
    body_height_m: float = None,
    body_weight_kg: float = None,
    sr_config: AnaropiaSRDataConfig = None,
    preproc_config: AnaropiaPreprocessingConfig = None,
    *,
    com: np.ndarray = None,
    stimulus: np.ndarray = None,
    name: str = None,
    # Deprecated keyword — accepted but ignored when sr_config is provided
    config: "AnaropiaPreprocessingConfig | None" = None,
):
    """
    Run a CSMI (Continuous Sensory Manipulation Identification) analysis for one trial.

    Loads the stimulus and COM response from *filename*, constructs an
    :class:`~balancepy.data_class.sr_data` object, fits the Peterka18 model, and
    returns the fitted model object.

    Parameters
    ----------
    filename : str, optional
        Path to the Anaropia CSV data file. Not required when *com* and
        *stimulus* are supplied directly.
    body_height_m : float
        Subject body height in metres (needed for COM calculation).
    body_weight_kg : float
        Subject body weight in kilograms (needed for model fitting).
    sr_config : AnaropiaSRDataConfig, optional
        Stimulus/response configuration. Defaults to ``SR_LEGACY_AP``.
    preproc_config : AnaropiaPreprocessingConfig, optional
        Preprocessing configuration. Defaults to ``AnaropiaPreprocessingConfig()``.
    com : np.ndarray, optional
        Pre-computed COM array (bypasses file loading when paired with *stimulus*).
    stimulus : np.ndarray, optional
        Pre-computed stimulus array.
    name : str, optional
        Label forwarded to ``sr_data.name``. If None, derived from *filename*.

    Returns
    -------
    subj : Peterka18
        Fitted Peterka18 model object with ``.params``, ``.fit_output``, and ``.plot()``.

    Examples
    --------
    >>> preproc_config = bp.AnaropiaPreprocessingConfig(end_time_seconds=220, cut_to_cycles=True)
    >>> subj = bp.run_csmi(filename, body_height_m=1.75, body_weight_kg=70,
    ...                    sr_config=bp.SR_LEGACY_AP, preproc_config=preproc_config)
    >>> subj.plot()
    """
    from balancepy.model_sim.peterka18 import Peterka18

    # --- Backward-compat: old-style `config` keyword ----------------------
    if config is not None and sr_config is None and preproc_config is None:
        warnings.warn(
            "Passing 'config' (AnaropiaPreprocessingConfig) to run_csmi "
            "is deprecated. Use sr_config + preproc_config instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        version = getattr(config, 'anaropia_version', 'legacy')
        sr_config = SR_LEGACY_AP if version == 'legacy' else SR_STANDARD_AP
        preproc_config = AnaropiaPreprocessingConfig(
            samplingrate_Hz=config.samplingrate_Hz,
            resample=config.resample,
            end_time_seconds=config.end_time_seconds,
            filter_type=config.filter_type,
            filter_order=config.filter_order,
            filter_cutoff_Hz=config.filter_cutoff_Hz,
            cut_to_cycles=config.cut_to_cycles,
            cycle_start_samples=config.cycle_start_samples,
            cycle_length_samples=config.cycle_length_samples,
        )

    if sr_config is None:
        sr_config = SR_LEGACY_AP
    if preproc_config is None:
        preproc_config = AnaropiaPreprocessingConfig()

    if com is None or stimulus is None:
        # Load from file
        raw_data = np.genfromtxt(filename, delimiter=',', names=True)
        time_raw = raw_data['time']

        _stim = _extract_stimulus(raw_data, sr_config)
        _com = _extract_response(raw_data, sr_config, body_height_m)

        _stim, _ = preprocess(_stim, time_raw, preproc_config)
        _com, _ = preprocess(_com, time_raw, preproc_config)
    else:
        # Use pre-computed arrays; only cut to cycles if requested
        if preproc_config.cut_to_cycles:
            _com = ts.cut_to_cycles(com, preproc_config.cycle_start_samples,
                                    preproc_config.cycle_length_samples)
            _stim = ts.cut_to_cycles(stimulus, preproc_config.cycle_start_samples,
                                     preproc_config.cycle_length_samples)
        else:
            _com, _stim = com, stimulus

    if name is None:
        name = sr_config.name
    if name is None and filename is not None:
        _parts = Path(filename).parts
        name = str(Path(*_parts[-4:])) if len(_parts) >= 4 else filename

    data_exp = data_class.sr_data(
        samplingrate_Hz=preproc_config.samplingrate_Hz,
        stimulus=_stim,
        response=_com,
        frequency_selection=sr_config.frequency_selection,
        name=name,
    )

    subj = Peterka18(body_weight_kg, body_height_m, data_exp=data_exp)
    subj.fit()
    return subj


