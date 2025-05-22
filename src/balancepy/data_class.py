from typing import Optional
import numpy as np
from numpy.typing import NDArray
import numpy.lib.recfunctions as rfn
import balancepy as bp



class stimulus_response_data:
    """
    Class for handling balancepy stimulus-response data in the frequency domain.
    This class is used to calculate the frequency response function (FRF), gain, phase, and coherence of a system based on the input stimulus and output response data.
    It also provides methods for selecting specific frequencies from the data.
    Args:
        samplingrate_Hz (int): Sampling rate in Hz.
        stimulus (NDArray[np.number]): Input stimulus data.
        response (NDArray[np.number]): Output response data.
        frequency_selection (str or list): Type of frequency selection ('all', 'prts', 'double_prts', or a list of indices).
    Attributes:
        time (NDArray[np.number]): Time vector calculated from the stimulus data.
        freq (NDArray[np.number]): Frequency vector calculated from the stimulus and response data.
        stimulus_spectrum (NDArray[np.number]): Spectrum of the input stimulus data.
        response_spectrum (NDArray[np.number]): Spectrum of the output response data.
        frf: Returns the frequency response function (FRF) of the system.
        gain: Returns the gain of the frf.
        phase: Returns the phase of the frf.
        coherence: Returns the coherence of the stimulus-response data.
    """
    def __init__(self, samplingrate_Hz: int, stimulus: NDArray[np.number], response: NDArray[np.number]=None, frequency_selection='all'):
        assert isinstance(samplingrate_Hz, int) and samplingrate_Hz > 0, "Samplingrate must be a positive integer."
        if response is not None: assert stimulus.shape[0] == response.shape[0], "Stimulus and response must have the same number of samples."

        # assign inputs to class variables
        self.samplingrate_Hz = samplingrate_Hz
        self.frequency_selection = frequency_selection

        # Calculate time vector
        self.time = np.arange(0, len(stimulus)) / samplingrate_Hz

        # Initialize stimulus
        self.stimulus = TD(stimulus)
        # Calculate spectra of stimulus and select frequencies
        stimulus_spectrum,_,freq = bp.spectrum(stimulus, samplingrate_Hz)

        # select relevant frequencies and assign to class variables
        self.stimulus_spectrum = FD(self.select_frequencies(stimulus_spectrum))

        self.freq = self.select_frequencies(freq)

        # Initialize response
        if response is not None:
            self.response = TD(response)
            response_spectrum,_,_ = bp.spectrum(response, samplingrate_Hz)
            self.response_spectrum = FD(self.select_frequencies(response_spectrum))

    # definition of class properties; properties are only calculated when called
    @property
    def frf(self):
        assert isinstance(self.response_spectrum, FD), "No response data found for FRF calculation."
        return bp.frf(self.stimulus_spectrum, self.response_spectrum)
    
    @property
    def gain(self):
        assert isinstance(self.response_spectrum, FD), "No response data found for FRF calculation."
        return abs(self.frf)
    
    @property
    def phase(self):
        assert isinstance(self.response_spectrum, FD), "No response data found for FRF calculation."
        return bp.phase(self.frf, self.freq)
    
    @property
    def coherence(self):
        assert isinstance(self.response_spectrum, FD), "No response data found for FRF calculation."
        assert self.response_spectrum.cycles.ndim == 2, "Multiple response cycles are required for coherence calculation."    

        return bp.coherence(self.stimulus_spectrum.cycles, self.response_spectrum.cycles)

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
        T = self.stimulus.cycles.shape[0] / self.samplingrate_Hz

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


class FD:
    def __init__(self, cycles: NDArray):
        self.cycles = cycles

    @property
    def average(self):
        return np.asarray(np.mean(self.cycles, axis=1))

    @property
    def PSD(self):
        avg = np.mean(self.cycles, axis=1)
        PSD = 1 / (self.samplingrate_Hz*2) * abs(avg)**2
        return np.asarray(np.sum(np.abs(self.cycles) ** 2))
    
    # def remnants(self):
    #     # Returns the difference between each cycle and the average cycle
    #     avg = np.mean(self.cycles, axis=1, keepdims=True)
    #     return self.cycles - avg

    def __getattr__(self, name):
        # Forward attribute access to the underlying array
        return getattr(self.cycles, name)

class TD:
    def __init__(self, cycles: NDArray):
        self.cycles = cycles

    @property
    def average(self):
        if self.cycles.ndim == 2:
            avg = np.mean(self.cycles, axis=1)
        else:
            avg = self.cycles
        return np.asarray(avg)

    @property
    def average_0(self):
        if self.cycles.ndim == 2:
            avg = np.mean(self.cycles, axis=1)
        else:
            avg = self.cycles
        return np.asarray(avg - np.mean(avg))

class simulation_data:
    """
    Class for handling balancepy simulation data in the frequency domain.
    This class is used to calculate the frequency response function (FRF), gain, phase, and coherence of a system based on the input stimulus and output response data.
    It also provides methods for selecting specific frequencies from the data.
    Attributes:
        samplingrate_Hz (int): Sampling rate in Hz.
        stimulus (NDArray[np.number]): Input stimulus data.
        response (NDArray[np.number]): Output response data.
        frequency_selection (str or list): Type of frequency selection ('all', 'prts', 'double_prts', or a list of indices).
        time (NDArray[np.number]): Time vector calculated from the stimulus data.
        freq (NDArray[np.number]): Frequency vector calculated from the stimulus and response data.
        stimulus_spectrum (NDArray[np.number]): Spectrum of the input stimulus data.
        response_spectrum (NDArray[np.number]): Spectrum of the output response data.
    Methods:
        frf: Returns the frequency response function (FRF) of the system.
        gain: Returns the gain of the system.
        phase: Returns the phase of the system.
        coherence: Returns the coherence of the system.
    """
    def __init__(self):
        self.samplingrate_Hz = None
        self.stimulus = None
        self.response = None
        self.frequency_selection = None
        self.time = None
        self.freq = None
        self.stimulus_spectrum = None
        self.response_spectrum = None
        self.frf = None
        self.coherence = None

    @property
    def gain(self):
        """Returns the gain of the system."""
        return np.asarray(abs(self.frf))
    
    @property
    def phase(self):
        """Returns the phase of the system."""
        return np.asarray(bp.phase(self.frf, self.freq))
