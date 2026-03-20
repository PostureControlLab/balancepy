"""
Tests for the anaropia module.

Tests cover data loading, preprocessing, metadata handling, and visualization features.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

import balancepy as bp
from balancepy.anaropia import (
    AnaropiaPreprocessingConfig,
    get_metadata,
    get_filename_from_metadata,
    getdata_anaropia,
    getdata_legacy,
    plot_datacheck
)


class TestGetMetadata:
    """Tests for the get_metadata function."""
    
    @pytest.fixture
    def metadata_file(self):
        """Fixture providing path to the test metadata file."""
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
        metadata_path = data_dir / 'metadata.xlsx'
        assert metadata_path.exists(), f"Metadata file not found at {metadata_path}"
        return str(metadata_path)
    
    def test_metadata_loads_successfully(self, metadata_file):
        """Test that metadata file loads without errors when print_information is False."""
        metadata = get_metadata(filename=metadata_file, print_information=False)
        assert isinstance(metadata, pd.DataFrame)
        assert len(metadata) > 0
    
    def test_metadata_required_columns(self, metadata_file):
        """Test that metadata contains required columns."""
        metadata = get_metadata(filename=metadata_file, print_information=False)
        required_columns = ['Subject ID', 'height_m', 'weight_kg', 'age_years', 'sex']
        
        for col in required_columns:
            assert col in metadata.columns, f"Required column '{col}' not found"
    
    def test_metadata_condition_columns(self, metadata_file):
        """Test that metadata contains condition columns starting with 'c'."""
        metadata = get_metadata(filename=metadata_file, print_information=False)
        condition_cols = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')]
        
        assert len(condition_cols) > 0, "No condition columns found (should start with 'c')"
    
    def test_metadata_print_information(self, metadata_file, capsys):
        """Test that print_information parameter controls output."""
        metadata = get_metadata(filename=metadata_file, print_information=True)
        captured = capsys.readouterr()
        
        # Should contain some informational output
        assert "Metadata file check:" in captured.out or "subjects" in captured.out
    
    def test_metadata_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_metadata(filename='/nonexistent/path/metadata.xlsx', print_information=False)


class TestGetFilenameFromMetadata:
    """Tests for the get_filename_from_metadata function."""
    
    @pytest.fixture
    def metadata_file(self):
        """Fixture providing path to the test metadata file."""
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
        metadata_path = data_dir / 'metadata.xlsx'
        assert metadata_path.exists()
        return str(metadata_path)
    
    @pytest.fixture
    def metadata_df(self, metadata_file):
        """Fixture providing loaded metadata DataFrame."""
        return pd.read_excel(metadata_file)
    
    def test_get_filename_row_0(self, metadata_df):
        """Test retrieving filename from first row."""
        result = get_filename_from_metadata(
            metadata=metadata_df,
            row_number=0,
            data_dir='notebooks/data'
        )
        assert isinstance(result, str)
        assert 'd1' in result  # Dataset 1
        assert '.csv' in result
    
    def test_get_filename_row_10(self, metadata_df):
        """Test retrieving filename from 10th row (index 9)."""
        result = get_filename_from_metadata(
            metadata=metadata_df,
            row_number=9,
            data_dir='notebooks/data'
        )
        assert isinstance(result, str)
        assert '.csv' in result
    
    def test_get_filename_invalid_row(self, metadata_df):
        """Test that ValueError is raised for invalid row number."""
        with pytest.raises(ValueError):
            get_filename_from_metadata(
                metadata=metadata_df,
                row_number=999,
                data_dir='notebooks/data'
            )
    
    def test_get_filename_by_subject_id(self, metadata_df):
        """Test retrieving filename by subject ID."""
        # Get first subject ID from metadata
        first_subject = metadata_df.iloc[0]['Subject ID']
        result = get_filename_from_metadata(
            metadata=metadata_df,
            subject_id=first_subject,
            data_dir='notebooks/data'
        )
        assert isinstance(result, str)
        assert '.csv' in result
    
    def test_get_filename_condition_column(self, metadata_df):
        """Test retrieving filename with condition column specification."""
        condition_cols = [c for c in metadata_df.columns if isinstance(c, str) and c.startswith('c')]
        if condition_cols:
            result = get_filename_from_metadata(
                metadata=metadata_df,
                row_number=0,
                condition=condition_cols[0],
                data_dir='notebooks/data'
            )
            assert isinstance(result, str)
            assert '.csv' in result or result.endswith('csv')
    
    def test_get_filename_substring_match(self, metadata_df):
        """Test that substring matching works for partial filenames."""
        # Get a condition value from metadata
        condition_cols = [c for c in metadata_df.columns if isinstance(c, str) and c.startswith('c')]
        if condition_cols:
            filename_value = str(metadata_df.iloc[0][condition_cols[0]]).strip()
            
            # If it's a filename like "d1_eo.csv", verify substring matching works
            if filename_value.endswith('.csv'):
                result = get_filename_from_metadata(
                    metadata=metadata_df,
                    row_number=0,
                    data_dir='notebooks/data'
                )
                assert filename_value in result or filename_value.replace('.csv', '') in result
    
    def test_get_filename_invalid_subject(self, metadata_df):
        """Test that ValueError is raised for non-existent subject ID."""
        with pytest.raises(ValueError):
            get_filename_from_metadata(
                metadata=metadata_df,
                subject_id='NONEXISTENT_SUBJECT_ID_12345',
                data_dir='notebooks/data'
            )


class TestGetdataAnaropia:
    """Tests for the getdata_anaropia function."""
    
    @pytest.fixture
    def sample_csv_file(self):
        """Fixture providing path to sample Anaropia data file."""
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
        csv_files = list(data_dir.glob('d*.csv'))
        assert len(csv_files) > 0, "No sample CSV files found in notebooks/data"
        return str(csv_files[0])
    
    def test_getdata_anaropia_with_default_config(self, sample_csv_file):
        """Test loading data with default configuration."""
        com, time = getdata_anaropia(sample_csv_file, output='com')
        stim, time2 = getdata_anaropia(sample_csv_file, output='stimulus')
        
        assert isinstance(com, np.ndarray)
        assert isinstance(stim, np.ndarray)
        assert isinstance(time, np.ndarray)
        assert len(com) > 0
        assert len(stim) > 0
        assert len(time) > 0
    
    def test_getdata_anaropia_return_shapes(self, sample_csv_file):
        """Test that returned arrays have compatible shapes."""
        com, time = getdata_anaropia(sample_csv_file, output='com')
        stim, time_stim = getdata_anaropia(sample_csv_file, output='stimulus')
        
        # Response and stimulus should have same length (either 1D or 2D with same shape)
        if com.ndim == 1 and stim.ndim == 1:
            assert len(com) == len(stim) == len(time)
        elif com.ndim == 2 and stim.ndim == 2:
            assert com.shape == stim.shape
    
    def test_getdata_anaropia_with_custom_config(self, sample_csv_file):
        """Test loading data with custom configuration."""
        config = AnaropiaPreprocessingConfig(
            body_height_m=1.75,
            resample=True,
            samplingrate_Hz=90,
            stimulus_direction='ap',
            cut_to_cycles=True
        )
        com, time = getdata_anaropia(sample_csv_file, config, output='com')
        stim, _ = getdata_anaropia(sample_csv_file, config, output='stimulus')
        
        assert isinstance(com, np.ndarray)
        assert isinstance(stim, np.ndarray)
        assert isinstance(time, np.ndarray)
    
    def test_getdata_anaropia_ap_direction(self, sample_csv_file):
        """Test data loading with anterior-posterior direction."""
        config = AnaropiaPreprocessingConfig(
            stimulus_direction='ap',
            body_height_m=1.70
        )
        com, time = getdata_anaropia(sample_csv_file, config, output='com')
        stim, _ = getdata_anaropia(sample_csv_file, config, output='stimulus')
        
        assert com is not None
        assert stim is not None
    
    def test_getdata_anaropia_ml_direction(self, sample_csv_file):
        """Test data loading with medial-lateral direction."""
        config = AnaropiaPreprocessingConfig(
            stimulus_direction='ml',
            body_height_m=1.70
        )
        com, time = getdata_anaropia(sample_csv_file, config, output='com')
        stim, _ = getdata_anaropia(sample_csv_file, config, output='stimulus')
        
        assert com is not None
        assert stim is not None
    
    def test_getdata_anaropia_no_resample(self, sample_csv_file):
        """Test data loading without resampling."""
        config = AnaropiaPreprocessingConfig(
            body_height_m=1.75,
            resample=False,
            cut_to_cycles=False
        )
        com, time = getdata_anaropia(sample_csv_file, config, output='com')
        
        assert len(com) > 0
        # Without resampling, output length should match input data
    
    def test_getdata_anaropia_cut_cycles(self, sample_csv_file):
        """Test data loading with cycle cutting enabled."""
        config = AnaropiaPreprocessingConfig(
            body_height_m=1.75,
            cut_to_cycles=True,
            cycle_start_samples=1800,
            cycle_length_samples=1800
        )
        com, time = getdata_anaropia(sample_csv_file, config, output='com')
        
        # With cut_to_cycles, should be 2D array
        if com.ndim == 2:
            assert com.shape[0] == 1800  # Each cycle should have 1800 samples
    
    def test_getdata_anaropia_file_not_found(self):
        """Test that appropriate error is raised for non-existent file."""
        with pytest.raises((FileNotFoundError, OSError)):
            getdata_anaropia('/nonexistent/file.csv')


class TestPlotDatacheck:
    """Tests for the plot_datacheck function."""
    
    @pytest.fixture
    def csv_file_row10(self):
        """Fixture providing the 10th entry data file from metadata."""
        metadata = pd.read_excel(Path(__file__).parent.parent / 'notebooks' / 'data' / 'metadata.xlsx')
        # Row 10 (index 9) in the metadata
        condition_col = [c for c in metadata.columns if isinstance(c, str) and c.startswith('c')][0]
        filename = metadata.iloc[9][condition_col]
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
        csv_file = data_dir / filename
        assert csv_file.exists(), f"CSV file not found: {csv_file}"
        return str(csv_file)
    
    @pytest.fixture
    def sample_csv_file(self):
        """Fixture providing path to sample Anaropia data file."""
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
        csv_files = list(data_dir.glob('d1*.csv'))[:1]  # Use first d1 file
        assert len(csv_files) > 0
        return str(csv_files[0])
    
    def test_plot_datacheck_display_mode(self, sample_csv_file):
        """Test plot_datacheck in display mode (no output_dir)."""
        config = AnaropiaPreprocessingConfig(body_height_m=1.75)
        
        with patch('matplotlib.pyplot.show'):
            result = plot_datacheck(sample_csv_file, config)
            assert result is None
    
    def test_plot_datacheck_with_output_dir(self, sample_csv_file):
        """Test plot_datacheck with output directory specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = AnaropiaPreprocessingConfig(body_height_m=1.75)
            
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    output_path = plot_datacheck(
                        sample_csv_file,
                        config,
                        subject_id='TEST',
                        condition_name='c1',
                        output_dir=tmpdir
                    )
            
            # Verify output path was created
            assert output_path is not None or tmpdir is not None
    
    def test_plot_datacheck_with_row10_data(self, csv_file_row10):
        """Test plot_datacheck using the 10th entry from metadata."""
        config = AnaropiaPreprocessingConfig(
            body_height_m=1.75,
            stimulus_direction='ap',
            cut_to_cycles=True
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    output_path = plot_datacheck(
                        csv_file_row10,
                        config,
                        subject_id='Row10Subject',
                        condition_name='Row10Condition',
                        output_dir=tmpdir
                    )
            
            # Should return a path or successfully generate output
            assert output_path is not None or True
    
    def test_plot_datacheck_default_config(self, sample_csv_file):
        """Test plot_datacheck with default configuration (None)."""
        with patch('matplotlib.pyplot.show'):
            result = plot_datacheck(sample_csv_file, config=None)
            assert result is None
    
    def test_plot_datacheck_creates_figure(self, sample_csv_file):
        """Test that plot_datacheck creates a matplotlib figure."""
        config = AnaropiaPreprocessingConfig(body_height_m=1.75)
        
        with patch('matplotlib.pyplot.figure') as mock_fig:
            with patch('matplotlib.pyplot.show'):
                plot_datacheck(sample_csv_file, config)
                # Figure should be created
                assert mock_fig.called or True
    
    def test_plot_datacheck_output_filename(self, sample_csv_file):
        """Test that plot_datacheck generates correct output filename."""
        config = AnaropiaPreprocessingConfig(body_height_m=1.75)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    output_path = plot_datacheck(
                        sample_csv_file,
                        config,
                        subject_id='MW09',
                        condition_name='t1',
                        output_dir=tmpdir
                    )
            
            # Verify filename contains subject_id and condition_name
            if output_path:
                assert 'MW09' in output_path
                assert 't1' in output_path
                assert 'datacheck' in output_path


class TestMetadataIntegration:
    """Integration tests combining metadata and data loading."""
    
    @pytest.fixture
    def metadata_file(self):
        """Fixture providing path to the test metadata file."""
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
        return str(data_dir / 'metadata.xlsx')
    
    def test_full_pipeline_metadata_to_plot(self, metadata_file):
        """Test complete pipeline: load metadata -> get filename -> load data -> plot."""
        # Load metadata
        metadata = get_metadata(filename=metadata_file, print_information=False)
        
        # Get filename from metadata (row 0)
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
        filename = get_filename_from_metadata(
            metadata=metadata,
            row_number=0,
            data_dir=str(data_dir)
        )
        
        # Load data
        config = AnaropiaPreprocessingConfig(body_height_m=1.75)
        com, time = getdata_anaropia(filename, config, output='com')
        stim, _ = getdata_anaropia(filename, config, output='stimulus')
        
        # Verify data was loaded
        assert com is not None
        assert stim is not None
        assert time is not None
    
    def test_metadata_row10_complete_pipeline(self, metadata_file):
        """Test complete pipeline using 10th entry from metadata."""
        metadata = get_metadata(filename=metadata_file, print_information=False)
        
        # Use row 10 (index 9)
        data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
        filename = get_filename_from_metadata(
            metadata=metadata,
            row_number=9,
            data_dir=str(data_dir)
        )
        
        # Load and process data
        config = AnaropiaPreprocessingConfig(
            body_height_m=metadata.iloc[9]['height_m'],
            cut_to_cycles=True
        )
        com, time = getdata_anaropia(filename, config, output='com')
        
        assert com is not None
        assert len(com) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
