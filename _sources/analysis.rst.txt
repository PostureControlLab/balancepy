Analysis
========

This module provides basic time-domain and frequency-domain analysis 
tools tailored for balance data. This comprises biomechanical 
calculations as well as signal processing methods.

.. autoclass:: balancepy.data_class.sr_data
   :undoc-members:
   :show-inheritance:

Attributes
~~~~~~~~~~

+-------------------------+---------------------------------------------------------------+
| Name                    | Description                                                   |
+=========================+===============================================================+
| samplingrate_Hz         | Sampling rate in Hz.                                          |
+-------------------------+---------------------------------------------------------------+
| stimulus                | Stimulus data in cycles (1D or 2D array).                     |
+-------------------------+---------------------------------------------------------------+
| response                | Response data in cycles (1D or 2D array).                     |
+-------------------------+---------------------------------------------------------------+
| frequency_selection     | Frequency selection method or indices.                        |
+-------------------------+---------------------------------------------------------------+
| name                    | Name of the data set.                                         |
+-------------------------+---------------------------------------------------------------+
| time                    | Time vector corresponding to the stimulus and response.       |
+-------------------------+---------------------------------------------------------------+
| freq                    | Frequencies corresponding to the spectra.                     |
+-------------------------+---------------------------------------------------------------+
| stimulus_spectrum       | Spectrum of the stimulus.                                     |
+-------------------------+---------------------------------------------------------------+
| response_spectrum       | Spectrum of the response.                                     |
+-------------------------+---------------------------------------------------------------+
| frf                     | Frequency response function.                                  |
+-------------------------+---------------------------------------------------------------+
| stimulus_mean           | Mean of the stimulus across cycles (property).                |
+-------------------------+---------------------------------------------------------------+
| response_mean           | Mean of the response across cycles (property).                |
+-------------------------+---------------------------------------------------------------+
| stimulus_mean0          | Mean of the stimulus, centered around 0 (property).           |
+-------------------------+---------------------------------------------------------------+
| response_mean0          | Mean of the response, centered around 0 (property).           |
+-------------------------+---------------------------------------------------------------+
| gain                    | Magnitude of the frequency response function (property).      |
+-------------------------+---------------------------------------------------------------+
| phase                   | Phase of the frequency response function (property).          |
+-------------------------+---------------------------------------------------------------+
| coherence               | Coherence between stimulus and response (property).           |
+-------------------------+---------------------------------------------------------------+
| stimulus_spectrum_mean  | Mean of the stimulus spectrum across cycles (property).       |
+-------------------------+---------------------------------------------------------------+
| response_spectrum_mean  | Mean of the response spectrum across cycles (property).       |
+-------------------------+---------------------------------------------------------------+
| stimulus_spectrum_PSD   | Power spectral density of the stimulus spectrum (property).   |
+-------------------------+---------------------------------------------------------------+
| response_spectrum_PSD   | Power spectral density of the response spectrum (property).   |
+-------------------------+---------------------------------------------------------------+

.. automodule:: balancepy.frequency
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: balancepy.timeseries
   :members:
   :undoc-members:
   :show-inheritance:


