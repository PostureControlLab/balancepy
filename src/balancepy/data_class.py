from typing import Optional
import numpy as np
from numpy.typing import NDArray
import balancepy as bp
from dataclasses import dataclass
from plotly.subplots import make_subplots
import plotly.graph_objects as go
@dataclass
class sr_data:
    """
    Class to hold stimulus and response data for balancepy models.

    Use `add_timedomain_data` to populate all attributes from stimulus and response data.

    Parameters
    ----------
    samplingrate_Hz : int, optional
        Sampling rate in Hz.
    stimulus : ndarray, optional
        Stimulus data in cycles (1D or 2D array).
    response : ndarray, optional
        Response data in cycles (1D or 2D array).
    frequency_selection : str or list, optional
        Method for frequency selection, e.g., 'all', 'prts', 'double_prts', or a list of indices.
    name : str, optional
        Name of the data set.
    """
    samplingrate_Hz: Optional[int] = None
    stimulus: Optional[NDArray] = None
    response: Optional[NDArray] = None
    frequency_selection: Optional[str] = 'all'
    name: Optional[str] = None

    @property
    def stimulus_mean(self):
        """
        Mean of the stimulus across cycles.

        Returns
        -------
        ndarray
            The mean stimulus.
        """
        assert self.stimulus is not None, "Stimulus must be set."
        if self.stimulus.ndim == 1:
            return self.stimulus
        elif self.stimulus.ndim == 2:
            return np.mean(self.stimulus, axis=1)

    @property
    def stimulus_mean0(self):
        """
        Mean of the stimulus, centered around zero.

        Returns
        -------
        numpy.ndarray
            The mean stimulus, centered around zero.
        """
        return self.stimulus_mean - np.mean(self.stimulus_mean)

    @property
    def response_mean(self):
        """
        Mean of the response across cycles.

        Returns
        -------
        ndarray
            The mean response.
        """
        assert self.response is not None, "Response must be set."
        if self.response.ndim == 1:
            return self.response
        elif self.response.ndim == 2:
            return np.mean(self.response, axis=1)

    @property
    def response_mean0(self):
        """
        Mean of the response, centered around zero.

        Returns
        -------
        numpy.ndarray
            The mean response, centered around zero.
        """
        return self.response_mean - np.mean(self.response_mean)

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

    def __post_init__(self):
        # Only run add_timedomain_data if all required arguments are provided
        if (
            self.samplingrate_Hz is not None and
            self.stimulus is not None and
            self.response is not None and
            self.frequency_selection is not None
        ):
            self.add_timedomain_data(
                samplingrate_Hz=self.samplingrate_Hz,
                stimulus=self.stimulus,
                response=self.response,
                frequency_selection=self.frequency_selection
            )

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
    


    def plot(data, fig=None, line_name=None, line=None):
        """
        Plot Bode diagram of a sr_data object using Plotly.

        Parameters
        ----------
        data : sr_data
            The sr_data object containing the stimulus and response data.
        fig : plotly.graph_objects.Figure, optional
            An existing figure to which the traces will be added. If None, a new figure will be created.
        line_name : str, optional
            Name for the line in the plot legend. If None, a default name will be used.
        line : dict, optional
            Dictionary containing line style properties (e.g., color, width). If None, a default color will be used.
        
        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure containing the Bode plot with time series, frequency response, and coherence plots.
        """
        
        if fig is None:
            fig = make_subplots(rows=3, cols=2, 
                        subplot_titles=("Stimulus Time Series", "Response Time Series", "Bode Magnitude Plot", "Coherence Plot", "Bode Phase Plot", "Parameters"),
                        specs=  [[{"type": "xy"}, {"type": "xy"}],
                                [{"type": "xy"}, {"type": "xy"}],
                                [{"type": "xy"}, {"type": "table"}]])


        if line is None:
            # Define a palette of 10 colors
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Check how many traces are in the subplot with title "Bode Magnitude Plot"
            n_traces = len([trace for trace in fig['data'] if trace['name'] != None])

            # Select the index for the color as n_traces % 10
            color_index = n_traces % 10

            line = dict(color=colors[color_index])

        # add name for legend
        if line_name is not None:
            pass
        elif data.name is not None:
            line_name = data.name
        else:
            line_name = f"Line {color_index + 1}"


        # Time domain plots
        if data.stimulus is not None:
            if data.stimulus.ndim==2:
                for i in range(data.stimulus.shape[1]):
                    fig.add_trace(go.Scatter(x=data.time, y=data.stimulus[:, i], 
                        mode='lines', line=dict(color='lightgray', width=1), 
                        name=None, showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.time, y=data.stimulus_mean, 
                        mode='lines', line=line, name=None, showlegend=False), 
                        row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=data.time, y=data.stimulus, 
                    mode='lines', line=line, name=None, showlegend=False), 
                    row=1, col=1)
            
        if data.response is not None:
            if data.response.ndim==2:
                for i in range(data.response.shape[1]):
                    fig.add_trace(go.Scatter(x=data.time, y=data.response[:, i], 
                        mode='lines', line=dict(color='lightgray', width=1), 
                        name=None, showlegend=False), row=1, col=2)
                    fig.add_trace(go.Scatter(x=data.time, y=data.response_mean, 
                        mode='lines', line=line, name=None, showlegend=False), 
                        row=1, col=2)
            else:
                fig.add_trace(go.Scatter(x=data.time, y=data.response, 
                    mode='lines', line=line, name=None, showlegend=False), 
                    row=1, col=2)

        # Frequency domain plots
        if data.frf is not None:
            fig.add_trace(go.Scatter(x=data.freq, y=data.gain, mode='lines', line=line, name=line_name, showlegend=True), row=2, col=1)
            fig.update_xaxes(type="log", row=2, col=1)
            fig.add_trace(go.Scatter(x=data.freq, y=data.phase, mode='lines', line=line, name=None, showlegend=False), row=3, col=1)
            fig.update_xaxes(type="log", row=3, col=1)

        if data.coherence is not None:
            fig.add_trace(go.Scatter(x=data.freq, y=data.coherence, mode='lines', line=line, name=None, showlegend=False), row=2, col=2)
            fig.update_xaxes(type="log", row=2, col=2)


        # if params_names is not None and params is not None:
        #     rounded_params = [round(param, 3) for param in params]
        #     fig.add_trace(go.Table(header=dict(values=params_names), cells=dict(values=np.array(rounded_params))), row=3, col=2)

        fig.update_layout(height=800, width=1000, title_text="Bode Plot")

        return fig