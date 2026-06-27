"""
Tests for the anaropia module.

Tests cover config dataclasses, helper functions, preprocessing pipeline,
data loading (deprecated API), and visualization features.
"""

import pytest
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import replace

import balancepy as bp
from balancepy.anaropia import (
    AnaropiaPreprocessingConfig,
    COMConfig,
    AnaropiaSRDataConfig,
    SR_LEGACY_AP,
    SR_LEGACY_ML,
    SR_STANDARD_AP,
    SR_STANDARD_ML,
    COM_LEGACY_AP,
    COM_LEGACY_ML,
    COM_STANDARD_AP,
    COM_STANDARD_ML,
    _extract_stimulus,
    _extract_response,
    preprocess,
    plot_datacheck,
    run_csmi,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv_file():
    """Path to a sample legacy-format Anaropia data file."""
    data_dir = Path(__file__).parent.parent / 'notebooks' / 'data'
    csv_files = list(data_dir.glob('d1*.csv'))
    assert len(csv_files) > 0, "No sample CSV files found in notebooks/data"
    return str(csv_files[0])


@pytest.fixture
def raw_data(sample_csv_file):
    """Loaded structured NumPy array from the sample CSV."""
    return np.genfromtxt(sample_csv_file, delimiter=',', names=True)


# ── AnaropiaPreprocessingConfig ──────────────────────────────────────────

class TestAnaropiaPreprocessingConfig:
    def test_defaults(self):
        cfg = AnaropiaPreprocessingConfig()
        assert cfg.samplingrate_Hz == 90
        assert cfg.resample is True
        assert cfg.filter_type is None
        assert cfg.cut_to_cycles is False

    def test_filter_type_validation(self):
        with pytest.raises(ValueError, match="filter_type"):
            AnaropiaPreprocessingConfig(filter_type='invalid_filter')

    def test_filter_type_none_allowed(self):
        cfg = AnaropiaPreprocessingConfig(filter_type=None)
        assert cfg.filter_type is None

    def test_replace(self):
        cfg = AnaropiaPreprocessingConfig()
        cfg2 = replace(cfg, resample=False, cut_to_cycles=False)
        assert cfg2.resample is False
        assert cfg2.cut_to_cycles is False
        assert cfg.resample is True  # original unchanged


# ── COMConfig ────────────────────────────────────────────────────────────

class TestCOMConfig:
    def test_legacy_ap(self):
        assert COM_LEGACY_AP.shoulder_pos_column == 'shld_zpos'
        assert COM_LEGACY_AP.hip_pos_column == 'hip_zpos'
        assert COM_LEGACY_AP.rotation is True

    def test_legacy_ml(self):
        assert COM_LEGACY_ML.shoulder_pos_column == 'shld_xpos'
        assert COM_LEGACY_ML.hip_pos_column == 'hip_xpos'

    def test_standard_ap(self):
        assert COM_STANDARD_AP.shoulder_pos_column == 'LeftShoulder_pos_z'

    def test_standard_ml(self):
        assert COM_STANDARD_ML.shoulder_pos_column == 'LeftShoulder_pos_x'


# ── AnaropiaSRDataConfig ────────────────────────────────────────────────

class TestAnaropiaSRDataConfig:
    def test_predefined_legacy_ap(self):
        assert SR_LEGACY_AP.stimulus_name == 'stim_pitch'
        assert SR_LEGACY_AP.response_name is COM_LEGACY_AP
        assert SR_LEGACY_AP.frequency_selection == 'prts'

    def test_predefined_legacy_ml(self):
        assert SR_LEGACY_ML.stimulus_name == 'stim_roll'

    def test_com_response_config(self):
        cfg = AnaropiaSRDataConfig(
            stimulus_name='stim_pitch',
            response_name=COM_LEGACY_AP,
        )
        assert cfg.response_name is COM_LEGACY_AP

    def test_custom_response_column(self):
        cfg = AnaropiaSRDataConfig(
            stimulus_name='stim_pitch',
            response_name='analog0',
        )
        assert cfg.response_name == 'analog0'



# ── _extract_stimulus / _extract_response ────────────────────────────────

class TestExtractHelpers:
    def test_extract_stimulus(self, raw_data):
        stim = _extract_stimulus(raw_data, SR_LEGACY_AP)
        assert isinstance(stim, np.ndarray)
        assert stim.ndim == 1
        assert len(stim) == len(raw_data)

    def test_extract_response_com(self, raw_data):
        resp = _extract_response(raw_data, SR_LEGACY_AP, body_height_m=1.75)
        assert isinstance(resp, np.ndarray)
        assert resp.ndim == 1
        assert len(resp) == len(raw_data['time'])

    def test_extract_response_com_requires_height(self, raw_data):
        with pytest.raises(ValueError, match="body_height_m"):
            _extract_response(raw_data, SR_LEGACY_AP, body_height_m=None)

    def test_extract_response_direct_column(self, raw_data):
        cfg = AnaropiaSRDataConfig(
            stimulus_name='stim_pitch',
            response_name='analog0',
        )
        resp = _extract_response(raw_data, cfg)
        assert isinstance(resp, np.ndarray)
        assert len(resp) == len(raw_data)


# ── preprocess() ─────────────────────────────────────────────────────────

class TestPreprocess:
    def test_full_pipeline(self, raw_data):
        signal = raw_data['stim_pitch']
        time = raw_data['time']
        cfg = AnaropiaPreprocessingConfig(cut_to_cycles=True)
        out, t = preprocess(signal, time, cfg)
        # Both signal and time are cut to cycles → 2D
        assert out.ndim == 2
        assert t.ndim == 2

    def test_no_resample_no_filter_no_cut(self, raw_data):
        signal = raw_data['stim_pitch']
        time = raw_data['time']
        cfg = AnaropiaPreprocessingConfig(resample=False, filter_type=None, cut_to_cycles=False)
        out, t = preprocess(signal, time, cfg)
        np.testing.assert_array_equal(out, signal)
        np.testing.assert_array_equal(t, time)

    def test_resample_only(self, raw_data):
        signal = raw_data['stim_pitch']
        time = raw_data['time']
        cfg = AnaropiaPreprocessingConfig(resample=True, filter_type=None, cut_to_cycles=False)
        out, t = preprocess(signal, time, cfg)
        assert out.ndim == 1
        expected_len = int(cfg.samplingrate_Hz * cfg.end_time_seconds)
        assert len(out) == expected_len


# ── plot_datacheck ───────────────────────────────────────────────────────

class TestPlotDatacheck:
    def test_default_config(self, sample_csv_file):
        fig, resp, stim = plot_datacheck(
            sample_csv_file, 1.75, save=False,
        )
        assert fig is not None
        assert isinstance(resp, np.ndarray)
        assert isinstance(stim, np.ndarray)

    def test_with_name(self, sample_csv_file):
        fig, _, _ = plot_datacheck(
            sample_csv_file, 1.75,
            sr_config=SR_LEGACY_AP,
            name='TEST_c1',
            save=False,
        )
        assert fig is not None

    def test_backward_compat_config_kwarg(self, sample_csv_file):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cfg = AnaropiaPreprocessingConfig()
            fig, _, _ = plot_datacheck(sample_csv_file, 1.75, config=cfg, save=False)
            assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
