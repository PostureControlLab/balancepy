# Project-layer functions for Anaropia balance analysis.
# These functions depend on ProjectPaths and operate across many files/subjects.
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Generator, Tuple
from dataclasses import dataclass
from .anaropia import (
    AnaropiaPreprocessingConfig,
    AnaropiaSRDataConfig,
    SR_LEGACY_AP,
    run_csmi,
    _get_column,
    _extract_response,
    preprocess,
)
from . import data_class


@dataclass
class ProjectPaths:
    """
    Fixed filesystem layout descriptor for an Anaropia analysis project.

    Encodes the standard directory convention::

        project_root/
            data/
                raw/                ← raw CSV files
                metadata.xlsx       ← subject/condition metadata
                quality_flags.csv   ← screening results
                processed/
                    sr_data/        ← pre-computed stimulus-response CSVs
            results/
                datacheck_plots/    ← PDF output of plot_datacheck()
                csmi_plots/         ← PDF output of run_csmi_batch(plot=True)

    Use this in a project-level ``src/config.py``::

        from pathlib import Path
        import balancepy as bp

        PROJECT_ROOT = Path(__file__).parent.parent.resolve()
        paths = bp.ProjectPaths(project_root=PROJECT_ROOT)

    ``Path(__file__)`` is immune to changes in the kernel working directory,
    so this is safe to import from any notebook regardless of where it was opened.

    Parameters
    ----------
    project_root : Path or str
        Absolute path to the project root.
    """

    project_root: Path
    data_raw_override: Path | str | None = None
    metadata_override: Path | str | None = None
    quality_flags_override: Path | str | None = None
    processed_dir_override: Path | str | None = None
    datacheck_plots_override: Path | str | None = None
    csmi_plots_override: Path | str | None = None

    def __post_init__(self):
        self.project_root = Path(self.project_root).resolve()
        self.data_raw_override = Path(self.data_raw_override).resolve() if self.data_raw_override is not None else None
        self.metadata_override = Path(self.metadata_override).resolve() if self.metadata_override is not None else None
        self.quality_flags_override = Path(self.quality_flags_override).resolve() if self.quality_flags_override is not None else None
        self.processed_dir_override = Path(self.processed_dir_override).resolve() if self.processed_dir_override is not None else None
        self.datacheck_plots_override = Path(self.datacheck_plots_override).resolve() if self.datacheck_plots_override is not None else None
        self.csmi_plots_override = Path(self.csmi_plots_override).resolve() if self.csmi_plots_override is not None else None

    @property
    def data_raw(self) -> Path:
        return self.data_raw_override if self.data_raw_override is not None else self.project_root / 'data' / 'raw'

    @property
    def metadata(self) -> Path:
        return self.metadata_override if self.metadata_override is not None else self.project_root / 'data' / 'metadata.xlsx'

    @property
    def quality_flags(self) -> Path:
        return self.quality_flags_override if self.quality_flags_override is not None else self.project_root / 'data' / 'quality_flags.csv'

    @property
    def processed_dir(self) -> Path:
        return self.processed_dir_override if self.processed_dir_override is not None else self.project_root / 'data' / 'processed' / 'sr_data'
    
    @property
    def datacheck_plots(self) -> Path:
        return self.datacheck_plots_override if self.datacheck_plots_override is not None else self.project_root / 'results' / 'datacheck_plots'

    @property
    def csmi_plots(self) -> Path:
        return self.csmi_plots_override if self.csmi_plots_override is not None else self.project_root / 'results' / 'csmi_plots'

# Quality flag utilities
def sync_quality_flags_from_metadata(paths: ProjectPaths) -> dict:
    """
    Ensure quality_flags.csv exists and has rows for every (subject_id, condition_name) in metadata.
    Existing rows are never modified. New rows get default values (flag_code=0, empty note).
    """
    if paths is None:
        raise ValueError("Pass 'paths' to sync_quality_flags_from_metadata.")

    from datetime import datetime

    metadata = get_metadata(paths, print_information=False)
    condition_cols = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')]
    if 'Subject ID' not in metadata.columns:
        raise ValueError("'Subject ID' column not found in metadata")

    # Precompute all files in data_raw for faster lookup
    all_files = [p for p in paths.data_raw.rglob('*') if p.is_file()]

    summary = {
        'created_csv': False,
        'rows_added': 0,
        'rows_existing': 0,
    }

    csv_cols = ['subject_id', 'condition_name', 'flag_code', 'trial_code', 'screened_at', 'note']

    # Check if quality_flags.csv exists; if not, create an empty DataFrame with required columns.
    if not paths.quality_flags.exists():
        flags_df = pd.DataFrame(columns=csv_cols)
        summary['created_csv'] = True
    else:
        flags_df = read_quality_flags(paths, normalize=False)

    now = datetime.now().isoformat(timespec='minutes')

    # Build set of existing keys for O(1) lookup
    existing_keys = set(
        zip(flags_df['subject_id'].astype(str), flags_df['condition_name'].astype(str))
    )

    new_rows = []

    # Ensure full metadata coverage in quality flags.
    for _, row in metadata.iterrows():
        subject_id = str(row['Subject ID'])
        for condition_name in condition_cols:
            cell = row[condition_name]
            trial_code = str(cell)

            flag_code = None
            try:
                cell_value = float(cell)
            except (ValueError, TypeError):
                cell_value = None

            if cell_value == -1:
                flag_code = 3
            else:
                # Raise if the file is not found; do not handle this silently.
                get_filename_from_metadata(paths, subject_id, condition_name, all_files=all_files, metadata=metadata)

            if (subject_id, str(condition_name)) in existing_keys:
                key_mask = (
                    flags_df['subject_id'].astype(str) == subject_id
                ) & (
                    flags_df['condition_name'].astype(str) == str(condition_name)
                )
                # Row exists but metadata previously encoded -1 but now contains entry.
                if flag_code == 3 and cell_value != -1:
                    flags_df.loc[key_mask, 'flag_code'] = 0
                    flags_df.loc[key_mask, 'trial_code'] = trial_code
                    flags_df.loc[key_mask, 'screened_at'] = now
                    flags_df.loc[key_mask, 'note'] = flags_df.loc[key_mask, 'note'].apply(
                        lambda n: (
                            f"{str(n).strip()} | " if pd.notna(n) and str(n).strip() else ""
                        ) + "Updated from missing file to found file"
                    )
                summary['rows_existing'] += 1
            else:
                new_rows.append({
                    'subject_id': subject_id,
                    'condition_name': condition_name,
                    'flag_code': flag_code,
                    'note': '',
                    'screened_at': now,
                    'trial_code': trial_code,
                })
                summary['rows_added'] += 1

    if new_rows:
        flags_df = pd.concat([flags_df, pd.DataFrame(new_rows)], ignore_index=True)

    write_quality_flags(flags_df, paths)
    return summary

def get_quality_flag_code(
    subject_id: str,
    condition_name: str,
    quality_flags: pd.DataFrame = None,
    paths: ProjectPaths = None,
) -> int | None:
    """Return the flag_code for (subject_id, condition_name), or None if not found."""
    
    if quality_flags is None:
        if paths is None:
            raise ValueError("Either 'quality_flags' or 'paths' must be provided.")
        quality_flags = read_quality_flags(paths)
    
    idx = quality_flags.index[
        (quality_flags['subject_id'].astype(str) == str(subject_id)) &
        (quality_flags['condition_name'].astype(str) == str(condition_name))
    ].tolist()
    
    if not idx:
        raise ValueError(f"No quality flag found for subject_id='{subject_id}', condition_name='{condition_name}'")
    if len(idx) > 1:
        raise ValueError(f"Multiple quality flags found for subject_id='{subject_id}', condition_name='{condition_name}'")
    
    # Get the scalar value directly
    val = quality_flags.at[idx[0], 'flag_code']
    if pd.isna(val):
        return None
    else:
        return int(val)

def _validate_quality_flags_df(
    df: pd.DataFrame,
    normalize: bool = True,
) -> pd.DataFrame:
    """Validate quality flag dataframe formatting, duplicates, and allowed codes."""
    required_cols = {'subject_id', 'condition_name', 'flag_code'}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(
            f"Missing required quality_flags columns: {missing_cols}. "
            f"Required: {sorted(required_cols)}"
        )

    duplicate_mask = df.duplicated(subset=['subject_id', 'condition_name'], keep=False)
    if duplicate_mask.any():
        duplicate_rows = (
            df.loc[duplicate_mask, ['subject_id', 'condition_name']]
            .drop_duplicates()
            .sort_values(['subject_id', 'condition_name'])
        )
        raise ValueError(
            "Duplicate quality flag rows found for (subject_id, condition_name): "
            f"{duplicate_rows.to_dict('records')}"
        )

    flag_numeric = pd.to_numeric(df['flag_code'], errors='raise').astype('Int64')
    invalid_mask = ~(flag_numeric.isin([0, 1, 2, 3]) | flag_numeric.isna())
    if invalid_mask.any():
        invalid_values = sorted(flag_numeric[invalid_mask].dropna().unique().tolist())
        raise ValueError(
            f"Invalid flag_code values found: {invalid_values}. "
            "Allowed values are 0 (ok), 1 (warning), 2 (do not use), 3 (file not found), or missing."
        )

    if normalize:
        df = df.copy()
        df['flag_code'] = flag_numeric

    return df

def read_quality_flags(
    paths: ProjectPaths,
    normalize: bool = True,
) -> pd.DataFrame | None:
    """
    Load quality flags from CSV file and perform formatting checks.

    Parameters
    ----------
    paths : ProjectPaths
        Project paths object used to resolve quality_flags location.
    normalize : bool, default True
        If True: convert ``flag_code`` to numeric (Int64). Duplicate keys and
        invalid flag values are always validated regardless of this setting.

    Returns
    -------
    pd.DataFrame or None

    Raises
    ------
    ValueError
        If required columns are missing, duplicate
        ``(subject_id, condition_name)`` rows exist, ``flag_code`` cannot be
        converted to numeric, or invalid flag codes are present.

    """

    if not paths.quality_flags.exists():
        return None
    
    df = pd.read_csv(
        paths.quality_flags,
        dtype={'subject_id': str, 'condition_name': str, 'trial_code': str, 'note': str}
    )

    return _validate_quality_flags_df(df, normalize=normalize)

def write_quality_flags(
    df: pd.DataFrame,
    paths: ProjectPaths,
) -> None:
    """Write quality flags CSV after validating unique keys and allowed codes."""
    df = _validate_quality_flags_df(df, normalize=True)

    paths.quality_flags.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.quality_flags, index=False)

def screen_quality_flags(
    paths: ProjectPaths,
    plots_path: Path | str | None = None,
    plot_prefix: str = 'datacheck',
    mode: str = 'new',
) -> None:
    """
    Inline notebook screening of data quality flags.

    Displays PNG plots one at a time in the cell output. Type a single-digit
    flag (0-3) + Enter. A note prompt appears only when flag != 0.
    Results are saved to ``quality_flags.csv`` after every entry.

    Parameters
    ----------
    paths : ProjectPaths
        Project paths object.
    plots_path : Path or str or None
        Directory containing plot PNGs. Defaults to ``paths.datacheck_plots``.
    plot_prefix : str
        Filename prefix for globbing/parsing (default ``'datacheck'``).
    mode : str
        ``'new'`` | ``'flagged'`` | ``'unflagged'`` | ``'all'``
    """
    from datetime import datetime
    from IPython.display import display, Image, clear_output

    plots_path = Path(plots_path) if plots_path is not None else paths.datacheck_plots

    sync_quality_flags_from_metadata(paths)
    metadata = get_metadata(paths, print_information=False)

    condition_cols = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')]
    slug_to_condition = {
        col.replace(': ', '_').replace(' ', '_'): col
        for col in condition_cols
    }

    flags_df = read_quality_flags(paths, normalize=True)
    if flags_df is None:
        flags_df = pd.DataFrame(columns=['subject_id', 'condition_name', 'flag_code', 'note', 'screened_at', 'trial_code'])

    def _get_current(subject_id, condition_name):
        mask = (
            (flags_df['subject_id'].astype(str) == str(subject_id)) &
            (flags_df['condition_name'].astype(str) == str(condition_name))
        )
        if mask.any():
            r = flags_df[mask].iloc[0]
            fc = int(r['flag_code']) if pd.notna(r['flag_code']) else None
            note = str(r['note']) if pd.notna(r['note']) else ''
            return fc, note
        return None, ''

    def _upsert(subject_id, condition_name, flag_code, note):
        nonlocal flags_df
        mask = (
            (flags_df['subject_id'].astype(str) == str(subject_id)) &
            (flags_df['condition_name'].astype(str) == str(condition_name))
        )
        now = datetime.now().isoformat(timespec='seconds')
        meta_mask = metadata['Subject ID'].astype(str) == str(subject_id)
        cell = metadata.loc[meta_mask, condition_name].values[0]
        trial_code = f"t{int(float(cell))}"

        if mask.any():
            flags_df.loc[mask, 'flag_code'] = flag_code
            flags_df.loc[mask, 'screened_at'] = now
            flags_df.loc[mask, 'trial_code'] = trial_code
            if note:
                flags_df.loc[mask, 'note'] = note
        else:
            new_row = pd.DataFrame([{
                'subject_id': subject_id, 'condition_name': condition_name,
                'flag_code': flag_code, 'note': note,
                'screened_at': now, 'trial_code': trial_code,
            }])
            flags_df = pd.concat([flags_df, new_row], ignore_index=True)

    # ── Select files to screen ────────────────────────────────────────────
    all_plot_files = sorted(plots_path.glob(f'{plot_prefix}_*.png'))
    parseable = [
        (f, *_parse_plot_filename(f.stem, slug_to_condition, plot_prefix))
        for f in all_plot_files
    ]
    parseable = [(f, sid, cname) for f, sid, cname in parseable if sid is not None]

    if mode == 'new':
        to_screen = [(f, sid, c) for f, sid, c in parseable if _get_current(sid, c)[0] is None]
    elif mode == 'flagged':
        to_screen = [(f, sid, c) for f, sid, c in parseable if _get_current(sid, c)[0] in (1, 2, 3)]
    elif mode == 'unflagged':
        to_screen = [(f, sid, c) for f, sid, c in parseable if _get_current(sid, c)[0] == 0]
    else:
        to_screen = parseable

    total = len(to_screen)
    if not total:
        print(f"No files to screen in mode='{mode}'.")
        return

    print(f"{total} file(s) to screen.  0=ok  1=warn  2=error  3=missing\n")

    # ── Screening loop ────────────────────────────────────────────────────
    for idx, (plot_file, subject_id, condition_name) in enumerate(to_screen, start=1):
        clear_output(wait=True)
        display(Image(filename=str(plot_file)))

        current_flag, current_note = _get_current(subject_id, condition_name)
        default_flag = current_flag if current_flag is not None else 0
        info = f"[{idx}/{total}]  {subject_id}  |  {condition_name}"
        if current_flag is not None:
            info += f"  (current: {current_flag})"
        print(info)

        raw = input(f"Flag [{default_flag}]: ").strip()
        flag_code = int(raw) if raw else default_flag

        note = ''
        if flag_code != 0:
            prefill = current_note if current_note else ''
            note = input(f"Note [{prefill}]: ").strip()
            if not note:
                note = prefill

        _upsert(subject_id, condition_name, flag_code, note)
        write_quality_flags(flags_df, paths)

    clear_output(wait=True)
    print(f"Screening complete. {total} trial(s) reviewed.")

# Metadata utitlities
def get_metadata(paths: ProjectPaths, print_information: bool = True) -> pd.DataFrame:
    """
    Load and validate metadata from an Excel file.
    
    This function reads metadata containing subject information and experimental
    conditions, then performs several checks:
    - Validates that all required columns are present and properly formatted
    - Extracts condition information (columns starting with 'c')
    - Reports basic dataset summary information
    
    Parameters
    ----------
    paths : ProjectPaths
        Project paths object used to resolve metadata and quality flag locations.
    
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
    """

    metadata = pd.read_excel(paths.metadata)
    
    if print_information:
    
        # Define required metadata columns
        required_columns = ['Subject ID', 'body_height_m', 'body_weight_kg', 'age_years', 'sex']

        print(f"File: {paths.metadata}")

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

        qf = read_quality_flags(paths, normalize=True)

        if qf is None:
            print("\nQuality flag summary: no quality_flags.csv found yet.")
        else:
            print(f"\nQuality flag summary (per condition):")
            print(f"{'Condition':<22} {'Good (0)':>8} {'Warn (1)':>8} {'Error (2)':>9} {'Missing (3)':>11} {'Unrated':>8}")
            print("-" * 74)
            for condition in condition_names:
                codes = []
                for sid in metadata['Subject ID'].astype(str) if 'Subject ID' in metadata.columns else []:
                    fc = get_quality_flag_code(sid, condition, quality_flags=qf)
                    if fc is not None:
                        codes.append(fc)
                n_good = sum(1 for c in codes if c == 0)
                n_warn = sum(1 for c in codes if c == 1)
                n_error = sum(1 for c in codes if c == 2)
                n_missing = sum(1 for c in codes if c == 3)
                n_unrated = n_subjects - len(codes)
                print(f"{condition:<22} {n_good:>8} {n_warn:>8} {n_error:>9} {n_missing:>11} {n_unrated:>8}")

    return metadata

def get_filename_from_metadata(
    paths: ProjectPaths,
    subject: str | int = 0,
    condition: str | int = 1,
    all_files: list[Path] | None = None,
    metadata: pd.DataFrame | None = None,
) -> str:
    """
    Obtain the data filename from metadata.
    
    Retrieves a data filename from metadata by subject and condition. If the metadata
    cell contains just a number, looks for a file matching: _s{subject_id}_t{number}.csv
    
    Parameters
    ----------
    paths : ProjectPaths
        Project paths object used to resolve raw data file locations.
    subject : str or int, optional
        ``str`` → matched against the 'Subject ID' column (e.g. ``'MW09AB13'``).
        ``int`` → used as a zero-based row index (e.g. ``0`` for first subject).
    condition : str or int, optional
        ``str`` → exact or prefix match against condition column names
        (e.g. ``'c1: eyes open'``, ``'c1'``).
        ``int`` → condition number, matched as ``c{n}...`` (e.g. ``1`` → ``'c1: ...'``).
    all_files : list of Path or None, optional
        Precomputed list of all files in data_raw. If provided, uses this instead of scanning.
    
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
    >>> filepath = get_filename_from_metadata(paths, 'MW09AB13', 'c1: eyes open')
    >>> filepath = get_filename_from_metadata(paths, 'MW09AB13', 1)
    >>> filepath = get_filename_from_metadata(paths, 0, 'c1: eyes open')
    >>> filepath = get_filename_from_metadata(paths, 0, 1)
    """

    if metadata is None:
        metadata = get_metadata(paths, print_information=False)

    # Determine row index — int → row index, str → Subject ID lookup
    if isinstance(subject, int):
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

    # Get subject ID for pattern matching (from the extracted row)
    subject_id = str(metadata.iloc[row_idx]['Subject ID']).strip()


    # Determine condition column — int → c{n} prefix match, str → exact or prefix match
    all_condition_cols = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')]

    if isinstance(condition, str):
        if condition in metadata.columns:
            condition_matches = [condition]
        else:
            condition_matches = [c for c in all_condition_cols if c.startswith(condition)]
    else:
        condition_matches = [c for c in all_condition_cols if c.startswith(f'c{condition}:') or c == f'c{condition}']

    if not condition_matches:
        raise ValueError(f"Condition or condition number {condition} not found in metadata columns")
    if len(condition_matches) > 1:
        raise ValueError(f"Multiple columns match condition {condition}: {condition_matches}")

    condition_col = condition_matches[0]
        
    # Extract the metadata cell and compute filename.
    metadata_cell_value = metadata.iloc[row_idx][condition_col]
    cell_str = str(metadata_cell_value).strip()

    try:
        trial_number = int(float(cell_str))
    except (ValueError, TypeError):
        raise FileNotFoundError(
            f"Metadata cell for subject '{subject_id}', condition '{condition_col}' "
            f"has an invalid or empty value: '{cell_str}'. Expected a trial number or -1."
        )
    else:
        if trial_number == -1:
            raise FileNotFoundError(
                f"Metadata indicates missing file for subject '{subject_id}', condition '{condition_col}'."
            )

        filename_ending = f"{subject_id}_t{trial_number}.csv"
    

    # search for file ending with filename in paths.data_raw (recursively, in case of subdirectories)
    if all_files is not None:
        matches = [p for p in all_files if p.name.endswith(filename_ending)]
    else:
        matches = [
            p for p in paths.data_raw.rglob('*')
            if p.is_file() and p.name.endswith(filename_ending)
        ]
    if len(matches) == 1:
        filename = str(matches[0])
    elif len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple files ending with '{filename_ending}' found under '{paths.data_raw}' "
            f"for subject '{subject_id}', condition '{condition_col}': {matches}"
        )
    else:
        raise FileNotFoundError(
            f"File not found under '{paths.data_raw}' with suffix '{filename_ending}'. "
            f"Metadata value for subject '{subject_id}', condition '{condition_col}' was '{cell_str}'."
        )

    return filename

def batch_iterator(
    paths: ProjectPaths,
    subjects: Union[int, str, List[Union[int, str]], None] = None,
    conditions: Union[int, str, List[Union[int, str]], None] = None,
    skip_flags: list | None = [2, 3],
    output_columns: List[str] = ['body_height_m'],
) -> Generator[Tuple, None, None]:
    """
    Iterate over subject-condition combinations in metadata, yielding file paths.
    
    Yields one tuple per valid subject-condition pair, with flexible indexing:
    - Subjects: row number (int), subject ID (str), or list of either
    - Conditions: condition number (int, e.g. 1 for "c1: ..."), full name (str), or list of either
    
    Missing files indicated by -1 in the metadata cell are skipped silently without warnings.
    
    Parameters
    ----------
    paths : ProjectPaths
        Project paths object used to resolve raw data and quality flag file paths.
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
    skip_flags : list of int or None, optional
        Flag codes that cause a trial to be skipped. Default ``[2, 3]`` skips
        error-rated and file-not-found trials.
        Set to ``None`` to yield all trials regardless of their flag code.
    output_columns : str or list[str] or tuple[str, ...] or None, optional
        Additional metadata columns to append to the yielded tuple in the order
        provided. Defaults to ``('body_height_m',)`` to preserve the current
        four-value output. Set to ``None`` or an empty sequence to yield only
        ``(subject_id, condition, filepath)``.

    Yields
    ------
    subject_id : str
        Subject identifier from metadata
    condition : str
        Full condition column name (e.g. 'c1: eyes open')
    filepath : str
        Full path to data file
    *extra_metadata
        Values from metadata columns listed in ``output_columns``.
    
    Notes
    -----
    Files marked with -1 in the metadata condition cell are treated as missing and are 
    skipped silently without warning messages.

    When ``quality_flags`` is provided (or loadable from ``paths.quality_flags``)
    and ``skip_flags`` is not None, trials whose flag_code is in ``skip_flags`` are
    silently skipped.
    
    """
    if paths is None:
        raise ValueError("Pass 'paths' to batch_iterator.")
    
    metadata = get_metadata(paths, print_information=False)

    # Normalize subjects to row indices
    subject_rows = _normalize_subject_selection(metadata, subjects)
    
    # Normalize conditions to column names
    condition_cols = _normalize_condition_selection(metadata, conditions)

    # Check for missing output columns in metadata
    missing_output_columns = [col for col in output_columns if col not in metadata.columns]
    if missing_output_columns:
        raise ValueError(
            f"Requested output_columns not found in metadata: {missing_output_columns}. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    # Resolve quality flags once for the whole iteration
    _qf = read_quality_flags(paths, normalize=True) if skip_flags is not None else None

    # Iterate over all combinations
    for row_idx in subject_rows:
        subject_id = metadata.iloc[row_idx]['Subject ID'] if 'Subject ID' in metadata.columns else f"row_{row_idx}"
        
        body_height_m = metadata.iloc[row_idx]['body_height_m']
        
        for condition_col in condition_cols:
            # Check if metadata cell value is -1 (indicates missing file)
            metadata_cell_value = metadata.iloc[row_idx][condition_col]
            try:
                metadata_cell_numeric = float(metadata_cell_value)
                if metadata_cell_numeric == -1:
                    continue  # Skip silently for missing files
            except (ValueError, TypeError):
                pass  # Not a numeric value, proceed with file lookup

            # Quality-flag based skipping
            if _qf is not None and skip_flags is not None:
                fc = get_quality_flag_code(str(subject_id), condition_col, _qf)
                if fc is not None and fc in skip_flags:
                    continue  # Skip silently

            filepath = get_filename_from_metadata(
                paths=paths,
                subject=row_idx,
                condition=condition_col
            )

            core = (subject_id, condition_col, filepath)
            if output_columns:
                extras = tuple(metadata.iloc[row_idx][column] for column in output_columns)
                yield core + extras
            else:
                yield core

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

def _parse_plot_filename(stem: str, slug_to_condition: dict, prefix: str = 'datacheck'):
    """Parse '{prefix}_{subject_id}_{condition_slug}' stem into (subject_id, condition_name).

    Returns (None, None) if the stem cannot be matched to a known condition slug.
    """
    full_prefix = f'{prefix}_'
    if not stem.startswith(full_prefix):
        return None, None
    rest = stem[len(full_prefix):]
    for slug, condition_name in slug_to_condition.items():
        if rest.endswith('_' + slug):
            subject_id = rest[: -(len(slug) + 1)]
            return subject_id, condition_name
    return None, None



# def _load_and_extract_sr_data(
#     filename: str,
#     sr_config: AnaropiaSRDataConfig,
#     body_height_m: float,
#     preproc_config: AnaropiaPreprocessingConfig,
#     use_file_prefix: bool = False,
# ) -> dict:
#     """
#     Internal helper: Load file and extract stimulus-response data pairs.
    
#     Parameters
#     ----------
#     filename : str
#         Path to CSV file.
#     sr_config : AnaropiaSRDataConfig
#         Stimulus/response configuration.
#     body_height_m : float
#         Subject height in meters.
#     preproc_config : AnaropiaPreprocessingConfig
#         Preprocessing configuration.
#     use_file_prefix : bool
#         If True, keys are (filename, stimulus_col). 
#         If False, keys are stimulus_col (original behavior).
    
#     Returns
#     -------
#     dict
#         Mapping of {key: sr_data} where key is either stimulus_col or (filename, stimulus_col).
#     """
#     raw_data = np.genfromtxt(filename, delimiter=',', names=True)
#     time_raw = raw_data['time']

#     # Extract and preprocess response (shared across all stimulus columns)
#     response_raw = _extract_response(raw_data, sr_config, body_height_m)
#     response, _ = preprocess(response_raw, time_raw, preproc_config)

#     # Normalize stimulus_name to a tuple
#     stim_cols = sr_config.stimulus_name
#     if isinstance(stim_cols, str):
#         stim_cols = (stim_cols,)

#     result = {}
#     for col in stim_cols:
#         stim_raw = _get_column(raw_data, col, sr_config)
#         stim, _ = preprocess(stim_raw, time_raw, preproc_config)
        
#         sr_obj = data_class.sr_data(
#             samplingrate_Hz=preproc_config.samplingrate_Hz,
#             stimulus=stim,
#             response=response,
#             frequency_selection=sr_config.frequency_selection,
#             name=f"{filename}:{col}" if use_file_prefix else col,
#         )
        
#         key = (filename, col) if use_file_prefix else col
#         result[key] = sr_obj

#     return result

# def _normalize_sr_data_items(
#     *,
#     filename: str = None,
#     sr_config: AnaropiaSRDataConfig = None,
#     body_height_m: float = None,
#     preproc_config: AnaropiaPreprocessingConfig = None,
#     configs: List[Union[tuple, dict]] = None,
# ) -> tuple[list[dict], bool]:
#     """Normalize single/multi SR inputs into canonical item dicts.

#     Returns
#     -------
#     items : list[dict]
#         Canonical items with keys: filename, sr_config, body_height_m, preproc_config.
#     use_file_prefix : bool
#         True when loading via multi-config mode.
#     """
#     default_preproc = preproc_config if preproc_config is not None else AnaropiaPreprocessingConfig()

#     if configs is None:
#         cfg = sr_config if sr_config is not None else SR_LEGACY_AP
#         if filename is None:
#             raise ValueError("'filename' is required when 'configs' is not provided.")
#         if body_height_m is None:
#             raise ValueError("'body_height_m' is required when 'configs' is not provided.")
#         return [
#             {
#                 'filename': filename,
#                 'sr_config': cfg,
#                 'body_height_m': body_height_m,
#                 'preproc_config': default_preproc,
#             }
#         ], False

#     if filename is not None or sr_config is not None:
#         raise ValueError("Do not pass 'filename' or 'sr_config' together with 'configs'.")

#     if not isinstance(configs, list) or len(configs) == 0:
#         raise ValueError("'configs' must be a non-empty list.")

#     items = []
#     for idx, item in enumerate(configs):
#         if isinstance(item, dict):
#             item_filename = item.get('filename', None)
#             item_sr_config = item.get('sr_config', None)
#             item_body_height = item.get('body_height_m', body_height_m)
#             item_preproc = item.get('preproc_config', default_preproc)
#         elif isinstance(item, tuple):
#             if len(item) == 2:
#                 item_filename, item_sr_config = item
#                 item_body_height = body_height_m
#                 item_preproc = default_preproc
#             elif len(item) == 3:
#                 item_filename, item_sr_config, item_body_height = item
#                 item_preproc = default_preproc
#             elif len(item) == 4:
#                 item_filename, item_sr_config, item_body_height, item_preproc = item
#             else:
#                 raise ValueError(
#                     f"configs[{idx}] tuple must have length 2, 3, or 4: "
#                     "(filename, sr_config[, body_height_m[, preproc_config]])."
#                 )
#         else:
#             raise TypeError(
#                 f"configs[{idx}] must be tuple or dict, got {type(item).__name__}."
#             )

#         if item_filename is None:
#             raise ValueError(f"configs[{idx}] is missing 'filename'.")
#         if item_sr_config is None:
#             raise ValueError(f"configs[{idx}] is missing 'sr_config'.")
#         if item_body_height is None:
#             raise ValueError(
#                 f"configs[{idx}] has no body height. Pass a global 'body_height_m' "
#                 "or include it in the item override."
#             )
#         if item_preproc is None:
#             item_preproc = AnaropiaPreprocessingConfig()

#         items.append(
#             {
#                 'filename': item_filename,
#                 'sr_config': item_sr_config,
#                 'body_height_m': item_body_height,
#                 'preproc_config': item_preproc,
#             }
#         )

#     return items, True

# def load_sr_data(
#     filename: str = None,
#     body_height_m: float = None,
#     sr_config: AnaropiaSRDataConfig = None,
#     preproc_config: AnaropiaPreprocessingConfig = None,
#     configs: List[Union[tuple, dict]] = None,
# ) -> dict:
#     """Load one or multiple Anaropia CSV files into ``sr_data`` objects.

#     Use either:
#     - single mode: ``filename`` + optional ``sr_config``
#     - multi mode: ``configs`` list with each item as tuple or dict

#     Multi-item tuple formats:
#     - ``(filename, sr_config)``
#     - ``(filename, sr_config, body_height_m)``
#     - ``(filename, sr_config, body_height_m, preproc_config)``

#     Multi-item dict format keys:
#     - required: ``filename``, ``sr_config``
#     - optional: ``body_height_m``, ``preproc_config``

#     Override precedence in multi mode:
#     - per-item value
#     - global function argument

#     Returns
#     -------
#     dict
#         Single mode: ``{stimulus_col: sr_data}``
#         Multi mode: ``{(filename, stimulus_col): sr_data}``
#     """
#     items, use_file_prefix = _normalize_sr_data_items(
#         filename=filename,
#         sr_config=sr_config,
#         body_height_m=body_height_m,
#         preproc_config=preproc_config,
#         configs=configs,
#     )

#     result = {}
#     for item in items:
#         sr_data_dict = _load_and_extract_sr_data(
#             filename=item['filename'],
#             sr_config=item['sr_config'],
#             body_height_m=item['body_height_m'],
#             preproc_config=item['preproc_config'],
#             use_file_prefix=use_file_prefix,
#         )
#         result.update(sr_data_dict)

#     return result

# def get_sr_data(
#     filename: str,
#     body_height_m: float,
#     sr_config: AnaropiaSRDataConfig = None,
#     preproc_config: AnaropiaPreprocessingConfig = None,
# ) -> dict:
#     """Load an Anaropia CSV and return one :class:`~balancepy.data_class.sr_data`
#     per stimulus column.

#     When ``sr_config.stimulus_name`` is a plain string a single-entry dict is
#     returned.  When it is a tuple of column names, one ``sr_data`` object is
#     built for each stimulus column (all sharing the same response signal).

#     For multi-file MIMO workflows, use :func:`load_sr_data` with ``configs``.

#     Parameters
#     ----------
#     filename : str
#         Path to the Anaropia CSV data file.
#     body_height_m : float
#         Subject body height in metres (needed for COM calculation).
#     sr_config : AnaropiaSRDataConfig, optional
#         Stimulus/response configuration. Defaults to ``SR_LEGACY_AP``.
#     preproc_config : AnaropiaPreprocessingConfig, optional
#         Preprocessing configuration. Defaults to
#         ``AnaropiaPreprocessingConfig()``.

#     Returns
#     -------
#     dict[str, sr_data]
#         Mapping from stimulus column name to the corresponding
#         :class:`~balancepy.data_class.sr_data` object.

#     Examples
#     --------
#     >>> sr_config = bp.AnaropiaSRDataConfig(
#     ...     ('stim_pitch', 'analog4'),
#     ...     response_name=bp.COM_LEGACY_AP,
#     ...     column_scales={'analog4': -1.0},
#     ... )
#     >>> sr_dict = bp.get_sr_data(filename, 1.75, sr_config=sr_config, preproc_config=config)
#     >>> sr_dict['stim_pitch'].plot()

#     See Also
#     --------
#     load_sr_data : Unified loader for single and multi-file workflows.
#     """
#     return load_sr_data(
#         filename=filename,
#         sr_config=sr_config,
#         body_height_m=body_height_m,
#         preproc_config=preproc_config,
#     )
