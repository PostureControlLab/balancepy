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
    body_height_m : float, default=0
        Height of subject in meters.
    resample : bool, default=True
        If True, resample data to samplingrate_Hz.
    samplingrate_Hz : int, default=90
        Desired sampling rate in Hz. 0 means no resampling. Used for standard Anaropia.
    stimulus_name : str, default='Screen'
        Name of the stimulus column in the data file. Use 'Screen' for standard Anaropia,
        'stim_pitch' for legacy Anaropia.
    stimulus_direction : str, default='ap'
        Direction of analysis: 'ap' (anterior-posterior) or 'ml' (medial-lateral).
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
    Standard Anaropia:
    
    >>> config = AnaropiaPreprocessingConfig(body_height_m=1.75, stimulus_direction='ap')
    >>> com, time = getdata_anaropia('data.csv', config, output='com')
    >>> stim, time = getdata_anaropia('data.csv', config, output='stimulus')
    
    Legacy Anaropia:
    
    >>> config = AnaropiaPreprocessingConfig(body_height_m=1.75, stimulus_name='stim_pitch', end_time_seconds=220)
    >>> com, time = getdata_legacy('legacy_data.csv', config, output='com')
    >>> stim, time = getdata_legacy('legacy_data.csv', config, output='stimulus')
    """

    body_height_m: float = 1.75
    resample: bool = True
    samplingrate_Hz: int = 90
    stimulus_name: str = 'stim_pitch'
    stimulus_direction: str = 'ap'
    cut_to_cycles: bool = True
    end_time_seconds: float = 260
    cycle_start_samples: int = 20*90
    cycle_length_samples: int = 20*90

    def __str__(self) -> str:
        """Pretty-print configuration settings."""
        settings = [
            f"body_height_m: {self.body_height_m}",
            f"resample: {self.resample}",
            f"samplingrate_Hz: {self.samplingrate_Hz}",
            f"stimulus_name: {self.stimulus_name}",
            f"stimulus_direction: {self.stimulus_direction}",
            f"cut_to_cycles: {self.cut_to_cycles}",
            f"end_time_seconds: {self.end_time_seconds}",
            f"cycle_start_samples: {self.cycle_start_samples}",
            f"cycle_length_samples: {self.cycle_length_samples}",
        ]
        return "AnaropiaPreprocessingConfig(\n  " + "\n  ".join(settings) + "\n)"


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
    subject_id: str | int | float | None = None,
    row_number: int | None = None,
    condition: str | None = None,
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
    subject_id : str or int or float, optional
        Subject identifier (e.g., 'MW09', 1, or 42). Can be a string or number.
        If None, uses first row.
    row_number : int, optional
        Row index in metadata DataFrame. Overrides subject_id if provided.
        If None and subject_id is None, uses first row (0).
    condition : str, optional
        Condition column name (e.g., 'c1: eyes open', 'c1', or just 'c1').
        If None, uses first condition column found (starting with 'c').
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
        If subject_id not found in metadata or row_number out of bounds.
    
    Examples
    --------
    >>> # Load specific subject (string or numeric)
    >>> filepath = get_filename_from_metadata(subject_id='MW09AB13', condition='c1')
    >>> filepath = get_filename_from_metadata(subject_id=42, condition='c1')
    >>> 
    >>> # Use specific row and condition
    >>> filepath = get_filename_from_metadata(row_number=0, condition='c1')
    >>> 
    >>> # Use defaults (first row, first condition)
    >>> filepath = get_filename_from_metadata()
    >>> 
    >>> # With custom metadata and data directory
    >>> filepath = get_filename_from_metadata(metadata=my_df, subject_id='MW09AB13', 
    ...                                       condition='c2', data_dir='data/raw')
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
    
    # Determine row index
    if row_number is not None:
        if row_number < 0 or row_number >= len(metadata):
            raise ValueError(f"row_number {row_number} out of bounds (metadata has {len(metadata)} rows)")
        row_idx = row_number
    elif subject_id is not None:
        if 'Subject ID' not in metadata.columns:
            raise ValueError("'Subject ID' column not found in metadata")
        
        # Convert both sides to string for comparison to handle mixed types
        # (e.g., numeric subject IDs in metadata vs. string input)
        subject_id_str = str(subject_id)
        metadata_subject_ids_str = metadata['Subject ID'].astype(str)
        matching_rows = metadata[metadata_subject_ids_str == subject_id_str]
        
        if matching_rows.empty:
            raise ValueError(f"Subject '{subject_id}' not found in metadata")
        row_idx = matching_rows.index[0]
    else:
        # Default to first row
        row_idx = 0
    
    # Determine condition column
    if condition is not None:
        # Try exact match first, then try to match by prefix
        if condition in metadata.columns:
            condition_col = condition
        else:
            # Try to find condition column that starts with the provided condition
            matching_cols = [c for c in metadata.columns 
                           if isinstance(c, str) and c.startswith(condition)]
            if matching_cols:
                condition_col = matching_cols[0]
            else:
                raise ValueError(f"Condition '{condition}' not found in metadata columns")
    else:
        # Use first condition column (starts with 'c')
        condition_cols = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')]
        if not condition_cols:
            raise ValueError("No condition columns found in metadata (columns starting with 'c')")
        condition_col = condition_cols[0]
    
    # Extract the cell value
    metadata_cell_value = metadata.iloc[row_idx][condition_col]
    
    # Get subject ID for pattern matching (from the extracted row)
    if 'Subject ID' in metadata.columns:
        row_subject_id = metadata.iloc[row_idx]['Subject ID']
    else:
        row_subject_id = subject_id if subject_id is not None else "unknown"
    
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



def getdata_anaropia(
    filename: str,
    config: AnaropiaPreprocessingConfig = None,
    output: str = 'com'
) -> tuple:
    """
    Access and format data from balance experiments recorded with Anaropia.

    Reads data recorded using the Anaropia virtual-reality application for 
    balance experiments. Calculates stimulus and center of mass (COM) data.

    Parameters
    ----------
    filename : str
        Path and filename to be analyzed.
    config : AnaropiaPreprocessingConfig, optional
        Configuration object containing processing parameters. If None, uses default
        configuration with standard settings.
    output : str, default='com'
        Specifies which data column to return alongside time.
        - 'com'      : Center of mass sway (computed from shoulder/hip positions).
        - 'stimulus' : Stimulus signal (column determined by config.stimulus_name
                       and config.stimulus_direction).
        - any other str : Raw column name from the data file (e.g. 'LeftShoulder_pos_z').

    Returns
    -------
    data : NDArray
        The requested data array (com, stimulus, or raw column).
    time : NDArray
        Time data.
    
    See Also
    --------
    AnaropiaPreprocessingConfig : Configuration class for Anaropia data preprocessing
    
    Examples
    --------
    >>> config = AnaropiaPreprocessingConfig(body_height_m=1.75, stimulus_direction='ap')
    >>> com, time = getdata_anaropia('data.csv', config, output='com')
    >>> stim, time = getdata_anaropia('data.csv', config, output='stimulus')
    """

    if config is None:
        config = AnaropiaPreprocessingConfig()

    # output_frequencies is a vector with the frequencies for which the FRF is calculated; default is up to 2 Hz
    # in case of the prts stimulus sequence, only every odd frequency point has energy, the even frequencies are zero

    data = np.genfromtxt(filename, delimiter=',', names=True)

    # --- Extract position data based on direction ---
    stimulus_name = config.stimulus_name
    if config.stimulus_direction == 'ap':
        sho = data['LeftShoulder_pos_z']
        hip = data['RightShoulder_pos_z']
        if stimulus_name == 'Screen':
            stimulus_name = 'Screen_rot_x'
    elif config.stimulus_direction == 'ml':
        if stimulus_name == 'Screen':
            stimulus_name = 'Screen_rot_z'
        sho = data['LeftShoulder_pos_x']
        hip = data['RightShoulder_pos_x']
    
    assert stimulus_name in data.dtype.names, f"Stimulus '{stimulus_name}' not found in data."

    sho_height = np.mean(data['LeftShoulder_pos_y'])
    hip_height = np.mean(data['RightShoulder_pos_y'])

    # --- Extract time and stimulus data ---
    time = data['Time']
    stimulus = data[stimulus_name]
    com = bm.get_com(sho, sho_height, hip, hip_height, config.body_height_m, True)

    # --- Select output column ---
    if output == 'com':
        result = com
    elif output == 'stimulus':
        result = stimulus
    else:
        assert output in data.dtype.names, f"Output column '{output}' not found in data."
        result = data[output]

    # --- Resample if requested ---
    if config.resample:
        result = ts.resample(time, result, config.samplingrate_Hz, config.end_time_seconds)
        time = ts.resample(time, time, config.samplingrate_Hz, config.end_time_seconds)

    # --- Cut to cycles if requested ---
    if config.cut_to_cycles:
        result = ts.cut_to_cycles(result, config.cycle_start_samples, config.cycle_length_samples)
        time = ts.cut_to_cycles(time, config.cycle_start_samples, config.cycle_length_samples)
    
    return result, time

def getdata_legacy(
    filename: str,
    config: AnaropiaPreprocessingConfig,
    output: str = 'com'
) -> tuple:
    """
    Access and format data from balance experiments recorded with Anaropia legacy.

    Reads data recorded using the legacy software version of Anaropia for 
    balance experiments. Calculates stimulus and center of mass (COM) data.

    Parameters
    ----------
    filename : str
        Path and filename to be analyzed.
    config : AnaropiaPreprocessingConfig
        Configuration object containing processing parameters. Must include body_height_m.
    output : str, default='com'
        Specifies which data column to return alongside time.
        - 'com'      : Center of mass sway (computed from shoulder/hip positions).
        - 'stimulus' : Stimulus signal (column given by config.stimulus_name).
        - any other str : Raw column name from the data file (e.g. 'shld_zpos').

    Returns
    -------
    data : NDArray
        The requested data array (com, stimulus, or raw column).
    time : NDArray
        Time data.
    
    See Also
    --------
    AnaropiaPreprocessingConfig : Configuration class for Anaropia data preprocessing
    
    Examples
    --------
    >>> config = AnaropiaPreprocessingConfig(body_height_m=1.75, stimulus_name='stim_pitch', end_time_seconds=220)
    >>> com, time = getdata_legacy('legacy_data.csv', config, output='com')
    >>> stim, time = getdata_legacy('legacy_data.csv', config, output='stimulus')
    """
    
    if config is None:
        raise ValueError("config must be provided for legacy data processing.")
    
    data = np.genfromtxt(filename, delimiter=',', names=True)

    # --- Extract time and stimulus data ---
    time = data['time']
    stimulus = data[config.stimulus_name]
    
    com = bm.get_com(
        data['shld_zpos'],
        np.mean(data['shld_ypos']),
        data['hip_zpos'],
        np.mean(data['hip_ypos']),
        config.body_height_m,
        True
    )

    # --- Select output column ---
    if output == 'com':
        result = com
    elif output == 'stimulus':
        result = stimulus
    else:
        assert output in data.dtype.names, f"Output column '{output}' not found in data."
        result = data[output]

    # --- Resample if requested ---
    if config.resample:
        result = ts.resample(time, result, config.samplingrate_Hz, config.end_time_seconds)
        time = ts.resample(time, time, config.samplingrate_Hz, config.end_time_seconds)

    # --- Cut to cycles if requested ---
    if config.cut_to_cycles:
        result = ts.cut_to_cycles(result, config.cycle_start_samples, config.cycle_length_samples)
        time = ts.cut_to_cycles(time, config.cycle_start_samples, config.cycle_length_samples)

    return result, time



def plot_datacheck(filename: str, config: AnaropiaPreprocessingConfig = None, save_fig: bool = True) -> str:
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
                   line=dict(width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_raw, y=raw_data['shld_zpos']-np.mean(raw_data['shld_zpos']), name='shoulder', mode='lines',
                   line=dict(width=1.5)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_raw, y=raw_data['hip_zpos']-np.mean(raw_data['hip_zpos']), name='hip', mode='lines',
                   line=dict(width=1.5)),
        row=1, col=1
    )
    fig.add_annotation(
        text=f"mean head: {np.mean(raw_data['zpos']):.4f}<br>mean sho: {np.mean(raw_data['shld_zpos']):.4f}<br>mean hip: {np.mean(raw_data['hip_zpos']):.4f}",
        xref="paper", yref="paper",
        x=1.14, y=0.99,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10),
        align="left",
        xanchor="right",
        yanchor="top"
    )
    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_yaxes(title_text="ap translation (m)", row=1, col=1)
    
    # --- PLOT 2: Stimulus and Stimulus Resampled ---
    # Get not resampled data not cut to cycles
    config_not_resampled = replace(config, resample=False, cut_to_cycles=False)
    response_not_resampled, time_not_resampled = bp.getdata_legacy(filename, config_not_resampled, output='com')
    stimulus_not_resampled, _ = bp.getdata_legacy(filename, config_not_resampled, output='stimulus')

    # Get resampled data not cut to cycles
    config_resampled = replace(config, resample=True, cut_to_cycles=False)
    response_resampled, time_resampled = bp.getdata_legacy(filename, config_resampled, output='com')
    stimulus_resampled, _ = bp.getdata_legacy(filename, config_resampled, output='stimulus')

    fig.add_trace(
        go.Scatter(x=time_not_resampled, y=stimulus_not_resampled, name='recorded sampling', 
                   mode='markers', marker=dict(size=1, color='blue'), 
                   opacity=0.7),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=time_resampled, y=stimulus_resampled, name='resampled', 
                   mode='markers', marker=dict(size=1, color='orange')),
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
    fig.update_yaxes(title_text=config.stimulus_name, row=2, col=1)
    
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
    response_cycles, time_cycles = getdata_legacy(filename, config, output='com')
    stimulus_cycles, _ = getdata_legacy(filename, config, output='stimulus')
    
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
        width=1122, height=794,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(x=1.02, y=0.9, xanchor="left", yanchor="top"),
        font=dict(size=12),
        title_font=dict(size=16),  # Override title
        xaxis_tickfont=dict(size=10),  # Override tick labels
        yaxis_tickfont=dict(size=10)  # Override tick labels
    )

    if save_fig := True:
        # Save figure to results/datacheck_plots folder
        output_dir = Path(filename).parent.parent.parent / "results" / "datacheck_plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"datacheck_{Path(filename).stem}.pdf"
        fig.write_image(str(output_file), width=561, height=397, scale=1)  # Save as PDF with high resolution


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
                    row_number=row_idx,
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
