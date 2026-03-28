# this submodule contains functions to analyze the data from anaropia balance experiments
import numpy as np
import balancepy as bp
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


def get_metadata(filename: str = None, print_information: bool = True) -> pd.DataFrame:
    """
    Load and validate metadata from an Excel file.
    
    This function reads metadata containing subject information and experimental
    conditions, then performs several checks:
    - Validates that all required columns are present and properly formatted
    - Extracts condition information (columns starting with 'c')
    - Reports data quality metrics for each condition using NFU (Not For Use) scores
    
    Parameters
    ----------
    filename : str, optional
        Path to metadata Excel file. If None, loads default file from 
        '../data/metadata.xlsx' relative to current working directory.
    
    Returns
    -------
    pd.DataFrame
        Metadata table with subject information and condition data.
    
    Raises
    ------
    Exception
        If no condition columns are found in the metadata file.
    
    Required Columns
    ----------------
    The metadata file must contain the following columns:
    - Subject ID : Unique identifier for each subject
    - body_height_m : Subject height in meters
    - body_weight_kg : Subject weight in kilograms
    - age_years : Subject age in years
    - sex : Subject sex (e.g., 'm', 'w')
    
    Column Naming Conventions
    -------------------------
    Condition columns : Start with 'c' (e.g., 'c1: eyes open', 'c2: eyes closed')
    NFU columns : Start with 'NFU ' followed by condition name (e.g., 'NFU c1: eyes open')
    
    NFU (Not For Use) Category Definitions
    ----------------------------------------
    Score 0 - All Good : The data file is complete, with no irregularities detected.
    Score 1 - Big Movements : Excessive participant movements were detected.
    Score 2 - >5 HMD Spikes : The data contains more than five spikes in the 
                              Head-Mounted Display (HMD) measurements.
    Score 3 - Other Irregularities : Any other issues not covered by the above categories 
                                     that might affect data quality.
    Score 4 - Poor FRF Fit : The Frequency Response Function (FRF) fit does not meet 
                             the required criteria.
    Score 5 - Marker Missing : One or more required markers are missing.
    Score 6 - File Missing : The corresponding data file is not available.
    
    Data Quality Grouping
    ----------------------
    - Valid (0): Ready for analysis
    - Warnings (1-3): Use with caution, review data quality
    - Invalid (4-6): Do not use for analysis
    """
        
    folder = Path.cwd().parent
    
    # --- Load metadata from file ---    
    if filename is not None:
        metadata = pd.read_excel(filename)
    elif (folder / "data" / "metadata.xlsx").exists():
        # Load from default location: parent_folder/data/metadata.xlsx
        metadata_file = folder / "data" / "metadata.xlsx"
        metadata = pd.read_excel(metadata_file)
    else:
        raise FileNotFoundError("Metadata file not found. Please provide a valid filename or ensure that '../data/metadata.xlsx' exists.")
    
    if print_information:
    
        # Define required metadata columns
        required_columns = ['Subject ID', 'body_height_m', 'body_weight_kg', 'age_years', 'sex']

        print(f"File: {metadata_file}")

        print("Metadata file check:")
        # --- Validate required columns ---
        has_warnings = False
        for col in required_columns:
            if col not in metadata.columns:
                print(f"WARNING: Required column '{col}' is missing.")
                has_warnings = True
            elif metadata[col].isna().all():
                print(f"WARNING: Required column '{col}' is empty or not properly formatted.")
                has_warnings = True
        
        if not has_warnings:
            print("✓ All required columns found and properly formatted.")


        # --- Extract and validate conditions ---
        condition_names = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')]
        
        if not condition_names:
            raise Exception("No conditions found in metadata file.")
        
        # --- Print summary statistics ---
        n_subjects = metadata['Subject ID'].nunique() if 'Subject ID' in metadata.columns else len(metadata)
        print(f"Number of subjects: {n_subjects}")
        print(f"\nCondition names and NFU labels:")
        
        # Print header for data quality table
        print(f"{'Condition':<18} {'Valid (0)':>10} {'Warnings (1-3)':>16} {'Invalid (4-6)':>16}")
        print("-" * 62)
        
        # Print data quality metrics for each condition
        for condition in condition_names:
            nfu_col = f"NFU {condition}"
            if nfu_col in metadata.columns:
                nfu = pd.to_numeric(metadata[nfu_col], errors="coerce")
                n_valid = (nfu == 0).sum()
                n_warn = nfu.between(1, 3, inclusive="both").sum()
                n_invalid = nfu.between(4, 6, inclusive="both").sum()
                print(f"{condition:<18} {n_valid:>10} {n_warn:>16} {n_invalid:>16}")
            else:
                print(f"{condition:<18} {'n/a':>10} {'n/a':>16} {'n/a':>16}")
        
    return metadata


def get_filename_from_metadata(
    metadata: pd.DataFrame | None = None,
    subject: str | int | None = None,
    condition: str | int | None = None,
    data_dir: str = 'data/raw'
) -> str:
    """
    Obtain the data filename from metadata, with intelligent fallback and fuzzy matching.
    
    Retrieves a data filename from metadata by subject and condition, with automatic
    defaults and fuzzy file matching. If the metadata cell contains just a number,
    searches for filenames matching: _s{subject_id}_t{number}.csv
    
    Parameters
    ----------
    metadata : pd.DataFrame, optional
        Metadata DataFrame. If None, loads default metadata file from '../data/metadata.xlsx'.
    subject : str or int, optional
        ``str`` → matched against the 'Subject ID' column (e.g. ``'MW09AB13'``).
        ``int`` → used as a zero-based row index (e.g. ``0`` for first subject).
        ``None`` → defaults to row 0.
    condition : str or int, optional
        ``str`` → exact or prefix match against condition column names
        (e.g. ``'c1: eyes open'``, ``'c1'``).
        ``int`` → condition number, matched as ``c{n}...`` (e.g. ``1`` → ``'c1: ...'``).
        ``None`` → uses the first condition column found.
    data_dir : str, default='data/raw'
        Directory where data files are located.
    
    Returns
    -------
    str
        Full path to the data file.
    
    Raises
    ------
    FileNotFoundError
        If metadata or data files not found, or if no matching file exists.
    ValueError
        If subject not found, row index out of bounds, or condition not found.
    
    Examples
    --------
    >>> filepath = get_filename_from_metadata(metadata, 'MW09AB13', 'c1: eyes open')
    >>> filepath = get_filename_from_metadata(metadata, 'MW09AB13', 1)
    >>> filepath = get_filename_from_metadata(metadata, 0, 'c1: eyes open')
    >>> filepath = get_filename_from_metadata(metadata, 0, 1)
    """
    from difflib import get_close_matches
    from pathlib import Path
    
    folder = Path.cwd().parent

    # Load metadata if not provided
    if metadata is None:
        if (folder / "data" / "metadata.xlsx").exists():
            metadata_file = folder / "data" / "metadata.xlsx"
            metadata = pd.read_excel(metadata_file)
        else:
            raise FileNotFoundError(
                "Metadata not provided and default metadata file '../data/metadata.xlsx' not found."
            )
    
    # Determine row index — int → row index, str → Subject ID lookup
    if subject is None:
        row_idx = 0
    elif isinstance(subject, int):
        if subject < 0 or subject >= len(metadata):
            raise ValueError(f"Row index {subject} out of bounds (metadata has {len(metadata)} rows)")
        row_idx = subject
    else:
        if 'Subject ID' not in metadata.columns:
            raise ValueError("'Subject ID' column not found in metadata")
        matching_rows = metadata[metadata['Subject ID'].astype(str) == str(subject)]
        if matching_rows.empty:
            raise ValueError(f"Subject '{subject}' not found in metadata")
        row_idx = matching_rows.index[0]

    # Determine condition column — int → c{n} prefix, str → exact/prefix match
    all_condition_cols = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')]
    if not all_condition_cols:
        raise ValueError("No condition columns found in metadata (columns starting with 'c')")

    if condition is None:
        condition_col = all_condition_cols[0]
    elif isinstance(condition, int):
        matching_cols = [c for c in all_condition_cols if c.startswith(f'c{condition}:') or c == f'c{condition}']
        if not matching_cols:
            raise ValueError(f"Condition number {condition} not found (looked for 'c{condition}...' in columns)")
        condition_col = matching_cols[0]
    else:
        if condition in metadata.columns:
            condition_col = condition
        else:
            matching_cols = [c for c in all_condition_cols if c.startswith(condition)]
            if not matching_cols:
                raise ValueError(f"Condition '{condition}' not found in metadata columns")
            condition_col = matching_cols[0]
    
    # Extract the cell value
    metadata_cell_value = metadata.iloc[row_idx][condition_col]
    
    # Get subject ID for pattern matching (from the extracted row)
    if 'Subject ID' in metadata.columns:
        row_subject_id = metadata.iloc[row_idx]['Subject ID']
    else:
        row_subject_id = str(subject) if subject is not None else "unknown"
    
    # Now use the original logic to find the file
    data_dir_path = (folder / data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist")
    
    # Convert metadata cell value to string
    cell_str = str(metadata_cell_value).strip()
    
    # Check if it's just a number (condition indicator)
    try:
        # If it's just a number, search for pattern _s{subject_id}_t{number}.csv
        condition_number = int(float(cell_str))
        search_pattern = f"_s{row_subject_id}_t{condition_number}.csv"
    except ValueError:
        # It's not just a number, so treat it as a potential filename
        search_pattern = cell_str
    
    # Try exact match first (with or without .csv)
    for attempt in [search_pattern, search_pattern + '.csv', search_pattern.replace('.csv', '')]:
        exact_path = data_dir_path / attempt
        if exact_path.exists():
            return str(exact_path)
    
    # No exact match - search for similar filenames
    available_files = [f.name for f in data_dir_path.iterdir() if f.is_file() and f.suffix == '.csv']
    
    if not available_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            f"Cannot locate data for subject '{row_subject_id}', condition '{condition_col}'."
        )
    
    # First try substring matching (search_pattern might be only part of the filename)
    substring_matches = [f for f in available_files if search_pattern in f]
    if substring_matches:
        # If multiple matches, return the first one
        matched_file = substring_matches[0]
        matched_path = data_dir_path / matched_file
        if matched_path.exists():
            return str(matched_path)
    
    # Find close matches (cutoff=0.6 means >60% similarity)
    similar = get_close_matches(search_pattern, available_files, n=3, cutoff=0.6)
    
    if similar:
        suggestions = '\n  - '.join(similar)
        raise FileNotFoundError(
            f"File '{search_pattern}' not found in '{data_dir}' "
            f"for subject '{row_subject_id}', condition '{condition_col}'.\n"
            f"Did you mean:\n  - {suggestions}"
        )
    else:
        # Show available files that match subject_id
        matching_subject = [f for f in available_files if str(row_subject_id) in f]
        if matching_subject:
            suggestions = '\n  - '.join(matching_subject[:5])
            raise FileNotFoundError(
                f"File '{search_pattern}' not found in '{data_dir}'.\n"
                f"Files found for subject '{row_subject_id}':\n  - {suggestions}"
            )
        else:
            raise FileNotFoundError(
                f"File '{search_pattern}' not found in '{data_dir}'.\n"
                f"No files found for subject '{row_subject_id}'. "
                f"First 5 available files: {', '.join(available_files[:5])}"
            )


def plot_datacheck(filename: str, body_height_m: float = None, config: AnaropiaPreprocessingConfig = None, save_fig: bool = True) -> str:
    """
    Plot data for comprehensive visual inspection of quality and structure.
    
    Generates a DinA4 landscape-formatted figure with multiple subplots for visually
    inspecting stimulus and response data using Plotly. The figure includes raw data translations,
    resampled stimulus and response with cycle indicators, and cycle-by-cycle analysis.
    
    Parameters
    ----------
    filename : str
        Path to the Anaropia data file (CSV format) to be plotted.
    config : AnaropiaPreprocessingConfig, optional
        Configuration object for data preprocessing. If None, uses default configuration.
    subject_id : str, optional
        Subject identifier for output filename. If None, uses input filename stem.
    condition_name : str, optional
        Condition name for output filename. If None, uses input filename stem.
    output_dir : str, optional
        Directory where the HTML will be saved. If None, displays the figure in browser.
    
    Returns
    -------
    str
        Path to the saved HTML file, or None if displayed instead of saved.
    
    Example
    -------
    >>> config = AnaropiaPreprocessingConfig(body_height_m=1.75)
    >>> plot_datacheck('data/raw/sMW09ON27_t1.csv', config, save_fig=True,
    ...                subject_id='MW09', condition_name='t1', 
    ...                output_dir='results/')
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from pathlib import Path
    from dataclasses import replace
    
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
                   mode='markers', marker=dict(size=1, color='blue'), 
                   opacity=0.7, legend='legend2'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_resampled, y=stimulus_resampled, name='resampled', 
                   mode='markers', marker=dict(size=1, color='orange'), legend='legend2'),
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
                   mode='markers', marker=dict(size=1, color='blue'), showlegend=False),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_resampled, y=response_resampled, name='resampled response', 
                   mode='markers', marker=dict(size=1, color='orange'), showlegend=False),
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
    response_cycles, time_cycles = getdata_anaropia(filename, output='com', body_height_m=body_height_m, config=config)
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
        title_text=f'Data Check: {Path(filename).stem}',
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

    if save_fig:
        # Save figure to results/datacheck_plots folder
        output_dir = Path(filename).parent.parent.parent / "results" / "datacheck_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"datacheck_{Path(filename).stem}.pdf"
        fig.write_image(str(output_file), width=794, height=562, scale=1)


    return fig



# ============================================================================
# Batch Processing Utilities
# ============================================================================

from typing import Union, List, Generator, Tuple


def batch_iterator(
    metadata: pd.DataFrame,
    subjects: Union[int, str, List[Union[int, str]], None] = None,
    conditions: Union[int, str, List[Union[int, str]], None] = None,
    data_dir: str = 'data/raw',
    skip_nfu: Union[List[int], None] = [4, 5, 6]
) -> Generator[Tuple[str, str, str, AnaropiaPreprocessingConfig], None, None]:
    """
    Iterate over subject-condition combinations in metadata, yielding file paths and configs.
    
    Yields one tuple per valid subject-condition pair, with flexible indexing:
    - Subjects: row number (int), subject ID (str), or list of either
    - Conditions: condition number (int, e.g. 1 for "c1: ..."), full name (str), or list of either
    
    Missing files indicated by -1 in the metadata cell are skipped silently without warnings.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table (typically from get_metadata()).
    subjects : int, str, list, or None, optional
        Subject(s) to process. Can be:
        - None: all subjects
        - int: single row number (0-indexed)
        - str: single subject ID
        - list: mix of row numbers and/or subject IDs, e.g. [0, 'MW09', 2]
    conditions : int, str, list, or None, optional
        Condition(s) to process. Can be:
        - None: all conditions
        - int: single condition number (1, 2, 3, ...)
        - str: full condition name (e.g. 'c1: eyes open')
        - list: mix of numbers and/or names, e.g. [1, 'c2: eyes closed']
    data_dir : str, optional
        Data directory relative to parent folder. Default: 'data/raw'
    skip_nfu : list of int or None, optional
        NFU (Not For Use) categories to skip. Entries with matching NFU values 
        will not be yielded. Convention: NFU columns are named "NFU {conditionname}".
        Default: [4, 5, 6] (Poor FRF Fit, Marker Missing, File Missing).
        Set to None to disable NFU filtering.
    
    Yields
    ------
    subject_id : str
        Subject identifier from metadata
    condition : str
        Full condition column name (e.g. 'c1: eyes open')
    filepath : str
        Full path to data file
    config : AnaropiaPreprocessingConfig
        Config with body_height_m set from metadata
    
    Notes
    -----
    Files marked with -1 in the metadata condition cell are treated as missing and are 
    skipped silently without warning messages.
    
    Entries with NFU (Not For Use) values in the skip_nfu list are also skipped silently.
    The NFU column is identified by the naming convention "NFU {conditionname}".
    By default, NFU categories 4, 5, and 6 are skipped (Poor FRF Fit, Marker Missing, File Missing).
    Set skip_nfu=None to disable this filtering.
    
    Examples
    --------
    >>> # All subjects, first 2 conditions (skips NFU 4, 5, 6 by default)
    >>> for sid, cond, fp, cfg in batch_iterator(metadata, conditions=[1, 2]):
    ...     plot_datacheck(fp, cfg)
    
    >>> # Specific subjects by ID, all conditions
    >>> for sid, cond, fp, cfg in batch_iterator(metadata, subjects=['MW09', 'AN16']):
    ...     plot_datacheck(fp, cfg)
    
    >>> # Row 0, condition by name, skip only NFU 6 (file missing)
    >>> for sid, cond, fp, cfg in batch_iterator(metadata, subjects=0, conditions='c1: eyes open', skip_nfu=[6]):
    ...     plot_datacheck(fp, cfg)
    
    >>> # Disable NFU filtering entirely
    >>> for sid, cond, fp, cfg in batch_iterator(metadata, subjects=[0, 'MW09'], conditions=[1, 2], skip_nfu=None):
    ...     # process each file
    ...     pass
    """
    
    # Normalize subjects to row indices
    subject_rows = _normalize_subject_selection(metadata, subjects)
    
    # Normalize conditions to column names
    condition_cols = _normalize_condition_selection(metadata, conditions)
    
    # Iterate over all combinations
    for row_idx in subject_rows:
        subject_id = metadata.iloc[row_idx]['Subject ID'] if 'Subject ID' in metadata.columns else f"row_{row_idx}"
        
        # Get body height for config if available
        body_height_m = metadata.iloc[row_idx].get('body_height_m', 1.75)  # default fallback
        
        for condition_col in condition_cols:
            # Check if metadata cell value is -1 (indicates missing file)
            metadata_cell_value = metadata.iloc[row_idx][condition_col]
            try:
                metadata_cell_numeric = float(metadata_cell_value)
                if metadata_cell_numeric == -1:
                    continue  # Skip silently for missing files
            except (ValueError, TypeError):
                pass  # Not a numeric value, proceed with file lookup
            
            # Check NFU (Not For Use) status if enabled
            if skip_nfu is not None:
                nfu_col_name = f"NFU {condition_col}"
                if nfu_col_name in metadata.columns:
                    try:
                        nfu_value = pd.to_numeric(metadata.iloc[row_idx][nfu_col_name], errors='coerce')
                        if pd.notna(nfu_value) and int(nfu_value) in skip_nfu:
                            continue  # Skip silently for NFU-marked entries
                    except (ValueError, TypeError):
                        pass  # If NFU value can't be converted, proceed
            
            try:
                # Get filepath using existing function
                filepath = get_filename_from_metadata(
                    metadata=metadata,
                    subject=row_idx,
                    condition=condition_col,
                    data_dir=data_dir
                )
                
                # Create config with subject-specific body height
                config = AnaropiaPreprocessingConfig()
                config.body_height_m = body_height_m
                
                yield subject_id, condition_col, filepath, config
                
            except (FileNotFoundError, ValueError) as e:
                # Skip this combination if file not found or other error
                print(f"Warning: Skipping {subject_id}, {condition_col}: {e}")
                continue


def _normalize_subject_selection(
    metadata: pd.DataFrame,
    subjects: Union[int, str, List[Union[int, str]], None]
) -> List[int]:
    """
    Convert subject specification to list of row indices.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table
    subjects : int, str, list, or None
        Subject specification
    
    Returns
    -------
    list of int
        Row indices to process
    
    Raises
    ------
    ValueError
        If subject not found or row number out of bounds
    TypeError
        If subject type not recognized
    """
    if subjects is None:
        # All rows
        return list(range(len(metadata)))
    
    # Normalize to list
    if isinstance(subjects, (int, str)):
        subjects = [subjects]
    
    row_indices = []
    for subject in subjects:
        if isinstance(subject, int):
            # Row number
            if 0 <= subject < len(metadata):
                row_indices.append(subject)
            else:
                raise ValueError(f"Row number {subject} out of bounds (metadata has {len(metadata)} rows)")
        elif isinstance(subject, str):
            # Subject ID
            if 'Subject ID' not in metadata.columns:
                raise ValueError("'Subject ID' column not found in metadata")
            matching = metadata[metadata['Subject ID'].astype(str) == subject].index
            if len(matching) == 0:
                raise ValueError(f"Subject '{subject}' not found in metadata")
            row_indices.append(matching[0])
        else:
            raise TypeError(f"Subject must be int or str, got {type(subject)}")
    
    return row_indices


def _normalize_condition_selection(
    metadata: pd.DataFrame,
    conditions: Union[int, str, List[Union[int, str]], None]
) -> List[str]:
    """
    Convert condition specification to list of column names.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table
    conditions : int, str, list, or None
        Condition specification
    
    Returns
    -------
    list of str
        Condition column names to process
    
    Raises
    ------
    ValueError
        If condition not found
    TypeError
        If condition type not recognized
    """
    # Get all condition columns (start with 'c')
    all_conditions = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')]
    
    if not all_conditions:
        raise ValueError("No condition columns found in metadata (columns starting with 'c')")
    
    if conditions is None:
        # All conditions
        return all_conditions
    
    # Normalize to list
    if isinstance(conditions, (int, str)):
        conditions = [conditions]
    
    condition_cols = []
    for condition in conditions:
        if isinstance(condition, int):
            # Condition number (1, 2, 3, ...)
            # Find column like "c{number}: ..."
            col_prefix = f"c{condition}"
            matching = [c for c in all_conditions if c.startswith(col_prefix + ':') or c == col_prefix]
            if len(matching) == 0:
                raise ValueError(f"Condition number {condition} not found (looked for 'c{condition}...' in columns)")
            condition_cols.append(matching[0])
        elif isinstance(condition, str):
            # Full condition name or partial match
            if condition in metadata.columns:
                condition_cols.append(condition)
            else:
                # Try to find by prefix
                matching = [c for c in all_conditions if c.startswith(condition)]
                if len(matching) == 0:
                    raise ValueError(f"Condition '{condition}' not found in metadata columns")
                condition_cols.append(matching[0])
        else:
            raise TypeError(f"Condition must be int or str, got {type(condition)}")
    
    return condition_cols


# ============================================================================
# CSMI Analysis
# ============================================================================

def run_csmi(
    filename: str,
    body_height_m: float,
    body_weight_kg: float,
    name: str = None,
    config: AnaropiaPreprocessingConfig = None
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

    com, _ = getdata_anaropia(filename, output='com', body_height_m=body_height_m, config=config)
    stim, _ = getdata_anaropia(filename, output='stimulus', config=config)

    if name is None:
        from pathlib import Path as _Path
        _parts = _Path(filename).parts
        name = str(_Path(*_parts[-4:])) if len(_parts) >= 4 else filename

    data_exp = data_class.sr_data(
        samplingrate_Hz=config.samplingrate_Hz,
        stimulus=stim,
        response=com,
        frequency_selection='prts',
        name=name
    )

    subj = Peterka18(body_weight_kg, body_height_m, data_exp=data_exp)
    subj.fit()
    return subj


def _run_csmi_job(
    subject_id: str,
    condition: str,
    filepath: str,
    body_height_m: float,
    body_weight_kg: float,
    config: AnaropiaPreprocessingConfig,
    plot: bool = False,
    overwrite_plots: bool = True
) -> dict:
    """
    Internal worker for parallel CSMI processing.

    Calls run_csmi and immediately extracts fitted parameters into a plain dict
    (required for safe inter-process serialisation with joblib).
    Column names follow the pattern ``{param_name}_{condition_slug}``
    where ``condition_slug`` replaces ': ' and ' ' with '_'.

    Returns
    -------
    dict
        Keys: 'Subject ID', one key per fitted parameter, 'fit_error_{condition_slug}'.
        On failure returns {'Subject ID': subject_id} with a warning printed.
    """
    slug = condition.replace(': ', '_').replace(' ', '_')
    try:
        subj = run_csmi(filepath, body_height_m, body_weight_kg, name=f"{subject_id}_{slug}", config=config)
        if plot:
            from pathlib import Path as _Path
            output_dir = _Path(filepath).parent.parent.parent / "results" / "csmi_plots"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"csmi_{subject_id}_{slug}.pdf"
            if overwrite_plots or not output_file.exists():
                fig = subj.plot()
                fig.write_image(str(output_file), width=397, height=562, scale=1)
        param_names = list(subj.params.names())
        param_vals = [float(x) for x in subj.params.values(only_free=False)]
        return {
            'Subject ID': subject_id,
            **{f'{name}_{slug}': val for name, val in zip(param_names, param_vals)},
            f'fit_error_{slug}': float(subj.fit_output.fun),
        }
    except Exception as e:
        print(f"  ✗ {subject_id} - {condition}: {e}")
        return {'Subject ID': subject_id}


def run_csmi_batch(
    metadata: pd.DataFrame,
    subjects: Union[int, str, List[Union[int, str]], None] = None,
    conditions: Union[int, str, List[Union[int, str]], None] = None,
    config: AnaropiaPreprocessingConfig = None,
    n_jobs: int = -1,
    plot: bool = False,
    overwrite_plots: bool = True
) -> pd.DataFrame:
    """
    Run CSMI analysis for all subjects and conditions in parallel.

    Returns a results DataFrame (Subject ID + fitted parameters) that can be
    saved independently and merged with metadata as needed. Fitted parameter
    columns follow the naming convention ``{param_name}_{condition_slug}``
    (e.g. ``W_c5_s0_v1``).

    Parameters
    ----------
    metadata : pd.DataFrame
        Metadata table as returned by get_metadata(). Used for file lookup and
        to extract body_height_m / body_weight_kg per subject.
    subjects : int, str, list, or None
        Subject selection passed to batch_iterator. None = all subjects.
    conditions : int, str, list, or None
        Condition selection passed to batch_iterator. None = all conditions.
    config : AnaropiaPreprocessingConfig, optional
        Shared preprocessing config for all trials. Defaults to
        AnaropiaPreprocessingConfig().
    n_jobs : int, default=-1
        Number of parallel workers (passed to joblib.Parallel).
        -1 uses all available CPUs.
    plot : bool, default=False
        If True, saves a Bode plot for each subject/condition as
        ``results/csmi_plots/csmi_{subject_id}_{condition_slug}.pdf``.
    overwrite_plots : bool, default=True
        If False, skips saving a plot when the PDF file already exists.
        Has no effect when ``plot=False``.

    Returns
    -------
    pd.DataFrame
        Results table with 'Subject ID' as key plus one column per fitted
        parameter per condition. One row per subject.

    Examples
    --------
    >>> config = bp.AnaropiaPreprocessingConfig()
    >>> config.end_time_seconds = 220
    >>> config.cut_to_cycles = True
    >>> csmi_df = bp.run_csmi_batch(metadata, conditions=['c5: s0_v1', 'c6: s0_v2'], config=config)
    >>> csmi_df.to_parquet('../results/csmi_results.parquet', index=False)
    >>> metadata = metadata.merge(csmi_df, on='Subject ID', how='left')
    """
    from joblib import Parallel, delayed

    if config is None:
        config = AnaropiaPreprocessingConfig()

    # kaleido (used for PDF export) spawns a Chromium subprocess per worker (~150-300 MB each).
    # Cap parallelism to 3 when plotting to avoid OOM; user can still pass a lower value explicitly.
    if plot and (n_jobs > 3 or n_jobs < 0):
        n_jobs = 3

    # Build job list in main process — only scalars cross process boundaries
    jobs = []
    for subject_id, condition, filepath, _ in batch_iterator(
        metadata, subjects=subjects, conditions=conditions, skip_nfu=None
    ):
        body_height_m = metadata.loc[metadata['Subject ID'] == subject_id, 'body_height_m'].values[0]
        body_weight_kg = metadata.loc[metadata['Subject ID'] == subject_id, 'body_weight_kg'].values[0]
        jobs.append((subject_id, condition, filepath, body_height_m, body_weight_kg, config))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_csmi_job)(*job, plot=plot, overwrite_plots=overwrite_plots) for job in jobs
    )

    # One result row per subject+condition; pivot to one row per subject
    return pd.DataFrame(results).groupby('Subject ID').first().reset_index()
