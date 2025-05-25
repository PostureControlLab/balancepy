from typing import Optional
import numpy as np
from numpy.typing import NDArray
import numpy.lib.recfunctions as rfn
import balancepy as bp
from dataclasses import dataclass, field

@dataclass
class sr_data:
    samplingrate_Hz: Optional[int] = None
    stimulus: Optional[NDArray] = None
    response: Optional[NDArray] = None
    frequency_selection: str = 'all'
    """
    A class to hold stimulus and response data for balancepy models.
    Use add_data method to retrieve all attributes from stimulus and response data.
    Attributes:
        samplingrate_Hz (Optional[int]): Sampling rate in Hz.
        time (Optional[NDArray]): Time vector corresponding to the stimulus and response.
        stimulus (Optional[NDArray]): Stimulus data in cycles.
        response (Optional[NDArray]): Response data in cycles.
        frequency_selection (str): Method for frequency selection, e.g., 'all', 'prts', 'double_prts'.
        freq (Optional[NDArray]): Frequencies corresponding to the spectra.
        stimulus_spectrum (Optional[NDArray]): Spectrum of the stimulus.
        response_spectrum (Optional[NDArray]): Spectrum of the response.
        frf (Optional[NDArray]): Frequency response function.
    """

    @property
    def stimulus_mean(self):
        """Returns the mean of the stimulus across cycles."""
        assert self.stimulus is not None, "Stimulus must be set."
        if self.stimulus.ndim == 1:
            return self.stimulus
        elif self.stimulus.ndim == 2:
            return np.mean(self.stimulus, axis=1)

    @property
    def stimulus_mean0(self):
        """Returns the mean of the stimulus, centered around 0."""
        return self.stimulus_mean - np.mean(self.stimulus_mean)

    @property
    def response_mean(self):
        """Returns the mean of the response across cycles."""
        assert self.response is not None, "Response must be set."
        if self.response.ndim == 1:
            return self.response
        elif self.response.ndim == 2:
            return np.mean(self.response, axis=1)

    @property
    def response_mean0(self):
        """Returns the mean of the response, centered around 0."""
        return self.response_mean - np.mean(self.response_mean)

    frequency_selection: str = 'all'
    freq: Optional[NDArray] = None
    stimulus_spectrum: Optional[NDArray] = None
    response_spectrum: Optional[NDArray] = None
    frf: Optional[NDArray] = None

    @property
    def gain(self):
        return abs(self.frf)
    
    @property
    def phase(self):
        return bp.phase(self.frf, self.freq)
    
    @property
    def coherence(self):
        if (self.response_spectrum is None or self.stimulus_spectrum is None
            or not self.response_spectrum.ndim == 2):
            return None
        else:
            return bp.coherence(self.stimulus_spectrum, self.response_spectrum)

    @property
    def stimulus_spectrum_mean(self):
        """Returns the mean of the stimulus spectrum across cycles."""
        assert self.stimulus_spectrum is not None, "Stimulus spectrum must be set."
        if self.stimulus_spectrum.ndim == 1:
            return self.stimulus_spectrum
        elif self.stimulus_spectrum.ndim == 2:
            return np.mean(self.stimulus_spectrum, axis=1)

    @property
    def response_spectrum_mean(self):
        """Returns the mean of the response spectrum across cycles."""
        assert self.response_spectrum is not None, "Response spectrum must be set."
        if self.response_spectrum.ndim == 1:
            return self.response_spectrum
        elif self.response_spectrum.ndim == 2:
            return np.mean(self.stimulus_spectrum, axis=1)

    @property
    def stimulus_spectrum_PSD(self):
        """Returns the power spectral density (PSD) of the stimulus spectrum.
            The PSD is scaled such that sum(PSD*df) = np.mean(stimulus_mean^2)"""
        assert self.samplingrate_Hz is not None, "Sampling rate must be set."
        assert self.stimulus_spectrum is not None, "Stimulus spectrum must be set."
        
        return 1 / (self.samplingrate_Hz*2) * abs(self.stimulus_spectrum_mean)**2

    @property
    def response_spectrum_PSD(self):
        """Returns the power spectral density (PSD) of the response spectrum.
            The PSD is scaled such that sum(PSD*df) = np.mean(response_mean^2)"""
        assert self.samplingrate_Hz is not None, "Sampling rate must be set."
        assert self.response_spectrum is not None, "Response spectrum must be set."
        
        return 1 / (self.samplingrate_Hz*2) * abs(self.response_spectrum_mean)**2

    # @property    
    # def remnants(self):
    #     # Returns the difference between each cycle and the average cycle
    #     avg = np.mean(self.cycles, axis=1, keepdims=True)
    #     return self.cycles - avg

    def add_timedomain_data(
        self,
        samplingrate_Hz: int,
        stimulus: NDArray[np.number],
        response: NDArray[np.number],
        frequency_selection: str = 'all'
    ):
        assert isinstance(samplingrate_Hz, int) and samplingrate_Hz > 0, "Samplingrate must be a positive integer."
        assert stimulus.shape[0] == response.shape[0], "Stimulus and response must have the same number of samples."
        
        self.samplingrate_Hz = samplingrate_Hz
        self.stimulus = stimulus
        self.response = response
        self.frequency_selection = frequency_selection
        
        self.time = np.arange(0, len(stimulus)) / samplingrate_Hz

        stimulus_spectrum, _, freq = bp.spectrum(stimulus, samplingrate_Hz)
        self.stimulus_spectrum = self.select_frequencies(stimulus_spectrum)
        self.freq = self.select_frequencies(freq)
        response_spectrum, _, _ = bp.spectrum(response, samplingrate_Hz)
        self.response_spectrum = self.select_frequencies(response_spectrum)
        self.frf = bp.frf(self.stimulus_spectrum, self.response_spectrum)


    def frequency_domain_recarray(self):
        """
        Returns a numpy recarray with all frequency domain outputs: freq, stimulus_spectrum, response_spectrum, frf, gain, phase, coherence.
        """
        dtype = [
            ('freq', self.freq.dtype),
            ('stimulus_spectrum', self.stimulus_spectrum.dtype),
            ('response_spectrum', self.response_spectrum.dtype),
            ('frf', self.frf.dtype),
            ('gain', self.gain.dtype),
            ('phase', self.phase.dtype),
            ('coherence', self.coherence.dtype)
        ]
        arr = np.rec.fromarrays(
            [
                self.freq,
                self.stimulus_spectrum.average,
                self.response_spectrum.average,
                self.frf,
                self.gain,
                self.phase,
                self.coherence
            ],
            dtype=dtype
        )
        return arr

    def time_domain_recarray(self):
        """
        Returns a numpy recarray with all time domain outputs: time, stimulus, response.
        """
        dtype = [
            ('time', self.time.dtype),
            ('stimulus', self.stimulus.dtype),
            ('response', self.response.dtype if self.response is not None else self.stimulus.dtype)
        ]
        arr = np.rec.fromarrays(
            [
                self.time,
                self.stimulus.average,
                self.response.average if self.response is not None else np.full_like(self.stimulus, np.nan)
            ],
            dtype=dtype
        )
        return arr


    def select_frequencies(self, data):
        """Returns data reduced to the selected frequencies.
        Args:
            data: data to be reduced (1D or 2D array)
        """
        type = self.frequency_selection

        # Get duration of the stimulus in seconds
        T = self.stimulus.shape[0] / self.samplingrate_Hz

        if isinstance(type, (list, np.ndarray)):
            selected_frequencies_index = np.array(type)
        elif type == 'all' or type is None:
            start = 0 
            end = int(round(2.5 * T)) # frequencies up to 2.5 Hz
            step = 1
            selected_frequencies_index = np.arange(start, end, step)
        elif type == 'prts':
            start = 0
            end = int(round(2.5 * T)) # frequencies up to 2.5 Hz
            step = 2
            selected_frequencies_index = np.arange(start, end, step)
        elif type == 'double_prts':
            start = 1
            end = int(round(2.5 * T)) # frequencies up to 2.5 Hz
            step = 4
            selected_frequencies_index = np.arange(start, end, step)
        else:
            raise ValueError("Unknown frequency selection type.")

        # Handle 1D or 2D data
        if data.ndim == 1:
            data = data[selected_frequencies_index]
        elif data.ndim == 2:
            data = data[selected_frequencies_index, :]
        else:
            raise ValueError("Data must be 1D or 2D array.")

        return data