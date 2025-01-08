import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import balancepy as bp

def bode_plot(FD, TD=None, legend=None, fig=None, params_names=None, params_values=None):
    """
    Plot Bode diagram of a time-domain model
    """
    if fig is None:
        fig = plt.figure()
    else:
        fig = fig

    if TD is not None:
        # Time domain plots
        plt.subplot(3, 2, 1)
        plt.plot(TD['time'], TD['stimulus_avg'])  # Stimulus time series
        plt.xlabel('time (s))')
        plt.ylabel('stimulus (°)')

        plt.subplot(3, 2, 2)
        plt.plot(TD['time'], TD['response_avg'])  # Response time series
        plt.xlabel('time (s)')
        plt.ylabel('response (°)')


    # Frequency domain plots
    plt.subplot(3, 2, 3)
    plt.semilogx(FD['freq'], FD['gain'])  # Bode magnitude plot
    plt.xlabel('frequency (Hz)')
    plt.ylabel('gain (°/°)')

    if 'coherence' in FD.dtype.names:
        plt.subplot(3, 2, 4)
        plt.semilogx(FD['freq'], FD['coherence'])  # Coherence plot
        plt.ylabel('coherence')
        plt.xlabel('frequency (Hz)')

    plt.subplot(3, 2, 5)
    plt.semilogx(FD['freq'], FD['phase'])  # Bode phase plot
    plt.ylabel('phase (°)')
    plt.xlabel('frequency (Hz)')

    if params_names is not None and params_values is not None:
        plt.subplot(3, 2, 6)
        plt.axis('off')
        table_data = list(zip(params_names, params_values))
        table = plt.table(cellText=table_data, colLabels=['Name', 'Value'], cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
    
    if legend is None and plt is not None:
        current_legend = plt.gca().get_legend_handles_labels()[1]
        i = len(current_legend)
        current_legend.extend(f'plot{i+1}')
        plt.legend(current_legend)
    elif legend is not None and plt is None:
        plt.legend(legend)
    elif legend is not None and plt is not None:
        current_legend = plt.gca().get_legend_handles_labels()[1]
        current_legend.append(legend)
        plt.legend(current_legend)    

    return plt
