import numpy as np
import pandas as pd
import scipy
import balancepy as bp
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def bode_plot(FD, TD=None, fig=None, line_name=None, params_names=None, params=None, line=None):
    """
    Plot Bode diagram of a time-domain model using Plotly
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
    if line_name is None:
        line_name = f"Line {color_index + 1}"


    if TD is not None:
        # Time domain plots
        fig.add_trace(go.Scatter(x=TD['time'], y=TD['stimulus_avg'], mode='lines', line=line, name=None, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=TD['time'], y=TD['response_avg'], mode='lines', line=line, name=None, showlegend=False), row=1, col=2)

    # Frequency domain plots
    fig.add_trace(go.Scatter(x=FD['freq'], y=FD['gain'], mode='lines', line=line, name=line_name, showlegend=True), row=2, col=1)
    fig.update_xaxes(type="log", row=2, col=1)

    if 'coherence' in FD.dtype.names:
        fig.add_trace(go.Scatter(x=FD['freq'], y=FD['coherence'], mode='lines', line=line, name=None, showlegend=False), row=2, col=2)
        fig.update_xaxes(type="log", row=2, col=2)

    fig.add_trace(go.Scatter(x=FD['freq'], y=FD['phase'], mode='lines', line=line, name=None, showlegend=False), row=3, col=1)
    fig.update_xaxes(type="log", row=3, col=1)

    if params_names is not None and params is not None:
        rounded_params = [round(param, 3) for param in params]
        fig.add_trace(go.Table(header=dict(values=params_names), cells=dict(values=np.array(rounded_params))), row=3, col=2)

    fig.update_layout(height=800, width=1000, title_text="Bode Plot")

    return fig
