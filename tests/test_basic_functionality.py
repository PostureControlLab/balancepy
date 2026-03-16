"""Unit tests for balancepy modules."""

import pytest
import numpy as np
import balancepy
from balancepy import make_stimulus, timeseries, frequency, biomechanics


class TestMakeStimulus:
    """Tests for stimulus generation functions."""
    
    def test_make_sine_basic(self):
        """Test basic sine wave generation."""
        stim = make_stimulus.make_sine(frequency_Hz=1, ncyc=1, ampl=1, samplingrate_Hz=1000)
        
        # Should have approximately 1000 samples for 1 cycle at 1 Hz with 1000 Hz sampling
        assert len(stim) == 1000
        assert isinstance(stim, np.ndarray)
        
    def test_make_sine_amplitude(self):
        """Test that sine amplitude is correct."""
        ampl = 5.0
        stim = make_stimulus.make_sine(frequency_Hz=1, ncyc=1, ampl=ampl, samplingrate_Hz=1000)
        assert np.max(np.abs(stim)) <= ampl + 1e-10  # Allow small numerical error
        
    def test_make_sine_frequency(self):
        """Test that sine frequency is correct."""
        frequency_Hz = 2
        ncyc = 5
        samplingrate_Hz = 1000
        stim = make_stimulus.make_sine(frequency_Hz=frequency_Hz, ncyc=ncyc, ampl=1, samplingrate_Hz=samplingrate_Hz)
        
        # Should have approximately ncyc * samplingrate_Hz / frequency_Hz samples
        expected_samples = int(ncyc * samplingrate_Hz / frequency_Hz)
        assert len(stim) == expected_samples


class TestTimeseries:
    """Tests for timeseries functions."""
    
    def test_cut_to_cycles_basic(self):
        """Test cutting data into cycles."""
        # Create simple 1D data
        data = np.arange(1000, dtype=float)
        cycle_length = 100
        
        cycles = timeseries.cut_to_cycles(data, cycle_start_samples=0, cycle_length_samples=cycle_length)
        
        # Should have correct shape
        assert cycles.shape[0] == cycle_length
        assert cycles.shape[1] == 10  # 1000 / 100 = 10 cycles
        
    def test_cut_to_cycles_with_discard(self):
        """Test discarding cycles."""
        data = np.arange(1000, dtype=float)
        cycle_length = 100
        discard = np.array([0, 9])  # Discard first and last cycle
        
        cycles = timeseries.cut_to_cycles(
            data, 
            cycle_start_samples=0, 
            cycle_length_samples=cycle_length,
            discard_cycles_index=discard
        )
        
        # Should have 8 cycles remaining (10 - 2)
        assert cycles.shape[1] == 8
        
    def test_resample_basic(self):
        """Test resampling functionality."""
        # Create simple time vector and data
        time_s = np.linspace(0, 1, 100)
        data = np.sin(2 * np.pi * time_s)
        new_sr = 200
        
        resampled = timeseries.resample(time_s, data, sampling_rate=new_sr)
        
        # Should have approximately correct length (endpoint=False means last sample excluded)
        # 1 second with 200 Hz sampling but endpoint=False gives 199 samples
        assert 199 <= len(resampled) <= 200
        assert isinstance(resampled, np.ndarray)


class TestFrequency:
    """Tests for frequency domain functions."""
    
    def test_spectrum_sine(self):
        """Test spectrum calculation on a sine wave."""
        # Create a simple sine wave
        fs = 1000  # Sampling rate
        f = 10     # Frequency
        t = np.arange(0, 1, 1/fs)
        data = np.sin(2 * np.pi * f * t)
        
        Sx, Sxx, freq = frequency.spectrum(data, fs)
        
        # Should have correct shapes
        assert len(Sx) == len(Sxx) == len(freq)
        assert len(freq) < len(data)  # Frequency vector is half the size
        
    def test_spectrum_output_types(self):
        """Test that spectrum returns correct types."""
        fs = 100
        data = np.random.randn(1000)
        
        Sx, Sxx, freq = frequency.spectrum(data, fs)
        
        assert isinstance(Sx, np.ndarray)
        assert isinstance(Sxx, np.ndarray)
        assert isinstance(freq, np.ndarray)


class TestBiomechanics:
    """Tests for biomechanics functions."""
    
    def test_winter_table_creation(self):
        """Test WinterTable instantiation."""
        height_m = 1.75
        mass_kg = 70
        
        # Create instance (takes only mass_kg and height_m)
        wt = biomechanics.WinterTable(mass_kg=mass_kg, height_m=height_m)
        
        assert wt.height_m == height_m
        assert wt.mass_kg == mass_kg
        assert hasattr(wt, 'J')
        assert hasattr(wt, 'smh_legs')
        
    def test_get_com_shape(self):
        """Test get_com output shape."""
        # Create simple input data
        shoulder_t = np.sin(np.linspace(0, 4*np.pi, 1000)) * 0.01
        hip_t = np.sin(np.linspace(0, 4*np.pi, 1000)) * 0.01
        
        com = biomechanics.get_com(
            shoulder_t=shoulder_t,
            shoulder_marker_height=1.5,
            hip_t=hip_t,
            hip_marker_height=0.9,
            height_m=1.75,
            rotation=True
        )
        
        # Should have same length as input
        assert len(com) == len(shoulder_t)
        assert isinstance(com, np.ndarray)


class TestPackageMetadata:
    """Tests for package metadata and configuration."""
    
    def test_version_string(self):
        """Test that version is a valid string."""
        assert isinstance(balancepy.__version__, str)
        assert len(balancepy.__version__) > 0
        
    def test_author_string(self):
        """Test that author information is present."""
        assert isinstance(balancepy.__author__, str)
        assert len(balancepy.__author__) > 0
