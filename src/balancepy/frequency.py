from numpy.typing import NDArray
import numpy.lib.recfunctions as rfn
import numpy as np
import skrf as rf
import balancepy as bp

def frequency_analysis(
    xi: NDArray[np.number],
    xo: NDArray[np.number],
    samplingrate: int,
    selected_frequencies = 'all',
    smoothing = None,
    bootstrap_samples: int = 0
    ) -> NDArray:
    """calculates frequency response functions (FRFs).

    Args:
        stim (NDArray[np.number]): 2D stimulus sequence with cycles in rows
        resp (NDArray[np.number]): 2D response data with cycles in rows
        samplingrate (float): sampling rate in samples/second
        selected_frequencies: 'all' or 'prts' or array of indices
        smoothing: None or function performing smoothing of the FRF

    Returns:
        NDArray[np.number]: matrix with frequency domain outputs
        f: frequency
        frf: frequency response function
        gain: gain of frequency response function
        pha: phase of frequency response function
        coh: coherence
    """

    yi,yii,f = spectrum(xi,samplingrate)
    yo,yoo,_ = spectrum(xo,samplingrate)
    
    if isinstance(selected_frequencies, np.ndarray):
        f   = f[selected_frequencies]
        yi  = yi[selected_frequencies,:]
        yo  = yo[selected_frequencies,:]
    elif selected_frequencies == 'prts': # selects every second frequency point up to 2 Hz
        selected_frequencies = np.arange(
            0,
            int(round(2 * np.size(xo, 0) / samplingrate)), 
            2
        )
        f   = f[selected_frequencies]
        yi  = yi[selected_frequencies,:]
        yo  = yo[selected_frequencies,:]
    elif selected_frequencies == 'double_prts': # selects every second frequency point up to 2 Hz
        selected_frequencies = np.arange(
            1,
            int(round(2 * np.size(xo, 0) / samplingrate)), 
            4
        )
        f   = f[selected_frequencies]
        yi  = yi[selected_frequencies,:]
        yo  = yo[selected_frequencies,:]
    elif selected_frequencies == 'all':
        f   = f
        yi  = yi
        yo  = yo

    # mean spectra
    yi_mean = abs(np.mean(yi,1))
    yo_mean = abs(np.mean(yo,1))
        
    # Calculate FRF, Magnitude and Phase of FRF, as well as Coherence
    # FRF from position data - Pintelon & Schoukens eq 2-17
    frf = bp.frf(yi, yo)
    coh = coherence(yi,yo)

    if smoothing is not None:
        f = smoothing(f, f)
        yi_mean = smoothing(yi_mean, f)
        yo_mean = smoothing(yo_mean, f)
        frf = smoothing(frf, f)
        coh = smoothing(coh, f)
    
    gain=abs(frf)
    pha=bp.phase(frf,f)        

    FD = rfn.merge_arrays([
            np.array(f,    dtype=[('freq','<f8')]),
            np.array(yi_mean, dtype=[('input_spectrum','<f8')]),
            np.array(yo_mean, dtype=[('output_spectrum','<f8')]),
            np.array(frf, dtype=[('frf','complex')]),
            np.array(gain, dtype=[('gain','<f8')]),
            np.array(pha,  dtype=[('phase','<f8')]),
            np.array(coh,  dtype=[('coherence','<f8')])
            ],
            flatten = True, usemask = False)

    if bootstrap_samples > 0:
        # Calculate confidence intervals using bootstrap
        # This is a placeholder for the actual bootstrap implementation
        print('Bootstrap confidence bound calculation not implemented yet')
        

    return FD


def spectrum(
    data: NDArray[np.number],
    sr: float,
) -> NDArray:
    """calculates properly scaled amplitude and power spectra of a time domain signal
        Spectra are calculated along columns of the input data array.
        Sx is scaled such that the amplitude of a sine input is given by abs(Sx)
        Sxx is scaled such that the integrated power density Sxx is equal to the 
        mean power of the time domain input. sum(Sxx*df) = mean(data^2). 
        df is the frequency bandwidth accounted for by each frequency point.

    Args:
        data (NDArray[np.number]): 1D or 2D data array to be resampled
        sr (float): sampling rate in samples/second

    Returns:
        NDArray[np.number]: scaled amplitude spectrum
        NDArray[np.number]: scaled power spectrum
        NDArray[np.number]: frequencies in Hz
    """
    
    N=np.size(data,0)              # number of samples in time axis
    f=np.arange(1,N/2+1) /N*sr  # frequency points for the output

    fk = np.fft.fft(data, axis=0)

    b= int(np.ceil(N/2)+1)
    y = fk[1:b,:]*2 # half sided spectrum

    Sx=1/N*y        # scaling to yield Sx, such that abs(Sx) = A

    Sxx = 1 / (sr*2*N) * abs(y)**2   # scaling to yield Sxx

    return Sx, Sxx, f

def coherence(yi,yo):
    """calculates coherence between two signals in the frequency domain

    Args:
        yi (NDArray[np.number]): input signal spectrum across cycles
        yo (NDArray[np.number]): output signal spectrum across cycles
        
    Returns:
        NDArray[np.number]: coherence
    """
    
    # calculate cross-power spectrum
    yoi = yo*np.conjugate(yi)

    yoi_mean=np.mean(yoi,1)
    yii_mean=np.mean(abs(yi)**2,1)
    yoo_mean=np.mean(abs(yo)**2,1)

    coh=(abs(yoi_mean)**2) / (yii_mean*yoo_mean)

    return coh

def frf(yi,yo):
    """calculates frequency response function (FRF) between two signals in the frequency domain

    Args:
        yi (NDArray[np.number]): input signal spectrum across cycles
        yo (NDArray[np.number]): output signal spectrum across cycles
        
    Returns:
        NDArray[np.number]: FRF
    """
            
    # Calculate cross-power spectrum
    yoi = yo * np.conjugate(yi)
    yii = yi * np.conjugate(yi)

    yoi_mean=np.mean(yoi,1)
    yii_mean=np.mean(yii,1)

    H = yoi_mean / yii_mean

    return H


def phase(frf,f):
    # calculate phase of frequency response function
    pha = np.angle(frf,deg=True)

    # create polynom roughly following a typical Phase curve of human sway responses + 180deg for modulo of 360deg
    # p_ref = 100-400*f+80*f**2 - 180
    p_ref = 20-100*f-30*f**2 - 180

    pha = np.mod(pha-p_ref,360) + p_ref
    
    return pha


def logspace(
        H: NDArray, 
        f: NDArray, 
        num_points: int = 20):
    """Interpolates the values of H to log-spaced frequencies.

    Returns 
    H_log NDArray
    f_log NDArray
    """
    from scipy.interpolate import CubicSpline
    from scipy.interpolate import PchipInterpolator
    from statsmodels.nonparametric.smoothers_lowess import lowess
    
    # Define the logarithmically spaced frequency axis:
    f_min = f[0]
    f_max = f[-1]
    f_log = np.logspace(np.log10(f_min), np.log10(f_max), num_points)

    # Use interpolation to get the values of y at the new log-spaced frequencies:
    # 
    # y_log = cubic_interpolator(f_log)

    G = abs(H)
    P = bp.phase(H, f)

    # G_interpolator = PchipInterpolator(f, G)
    # G_interpolator = CubicSpline(f, G)
    # G_log = G_interpolator(f_log)

    # P_interpolator = PchipInterpolator(f, P)
    # P_interpolator = CubicSpline(f, P)
    # P_log = P_interpolator(f_log)

    # Apply LOESS to magnitude and phase
    frac = 0.1  # span fraction, adjust as needed
    G_log = lowess(G, f, frac=frac, return_sorted=False)
    P_log = lowess(P, f, frac=frac, return_sorted=False)

    G_log = np.interp(f_log, f, G_log)
    P_log = np.interp(f_log, f, P_log)

    H_log = G_log * np.exp(1j * P_log/180*np.pi)

    return H_log, f_log



def logspace_averaging(frf, freq, n_points = 20):
    """
    Smooth a frequency response function (FRF) by averaging over 
    logarithmically spaced frequency bins using arithmetic averaging.
    
    Parameters
    ----------
    freq : array_like
        Input frequency vector (assumed strictly increasing and positive).
    frf : array_like
        Corresponding FRF values.
    n_points : int
        Desired number of output frequency bins.

    Returns
    -------
    freq_out : ndarray
        Output frequencies for non-empty bins (geometric mean of frequencies in each bin).
    frf_out : ndarray
        Averaged FRF values for the corresponding bins.
    """
    freq = np.array(freq)
    frf = np.array(frf)

    f_min = freq[0]
    f_max = freq[-1]
    bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_points + 1)

    freq_out_list = []
    frf_out_list = []

    for i in range(n_points):
        f_low = bin_edges[i]
        f_high = bin_edges[i+1]
        in_bin = (freq >= f_low) & (freq < f_high)

        # Only process non-empty bins
        if np.any(in_bin):
            freq_in_bin = freq[in_bin]
            frf_in_bin = frf[in_bin]
            # Arithmetic mean of FRF in this bin
            frf_mean = np.mean(frf_in_bin)
            # Geometric mean of frequencies in this bin
            f_geo_mean = np.mean(freq_in_bin)

            freq_out_list.append(f_geo_mean)
            frf_out_list.append(frf_mean)

    return np.array(frf_out_list), np.array(freq_out_list)



def logspace_vector(
        y: NDArray, 
        freq: NDArray, 
        num_points: int = 20):
    
    import scipy.signal as signal
    from scipy.optimize import basinhopping


    def objective(theta):
        #calculate model frequency response
        num = theta[:int(len(theta)/2)]
        den = theta[int(len(theta)/2):]
        
        tf = signal.TransferFunction(num, den)
        w, frf_sim = signal.freqresp(tf, w=freq*2*np.pi)
    
        #calculate objective
        err = np.sum( np.abs(frf_sim - y)**2 )

        return err
    
    theta_init = np.ones(12)
    minimizer_kwargs = {"method": "L-BFGS-B"}
    fit_output = basinhopping(objective, theta_init, minimizer_kwargs=minimizer_kwargs)

    theta_opt = fit_output.x
    num = theta_opt[:int(len(theta_opt)/2)]
    den = theta_opt[int(len(theta_opt)/2):]

    tf = signal.TransferFunction(num, den)
    f_log = np.logspace(np.log10(freq[0]), np.log10(freq[-1]), num_points)
    w, frf_log = signal.freqresp(tf, w=f_log*2*np.pi)

    y_log = frf_log

    return y_log, f_log


def logspace_manual_60s(x,f):
    if x.ndim == 1:
        reduced_x = np.array([
            x[0],                 # :,1
            x[1],                 # :,2
            np.mean(x[2:4]),      # :,3:4
            np.mean(x[3:5]),      # :,4:5
            np.mean(x[4:7]),      # :,5:7
            np.mean(x[5:9]),      # :,6:9
            np.mean(x[7:11]),     # :,8:11
            np.mean(x[9:13]),     # :,10:13
            np.mean(x[11:16]),    # :,12:16
            np.mean(x[15:20]),    # :,16:20
            np.mean(x[19:25]),    # :,20:25
            np.mean(x[24:32]),    # :,25:32
            np.mean(x[31:40]),    # :,32:40
            np.mean(x[39:49]),    # :,40:49
            np.mean(x[48:59]),    # :,49:59
            np.mean(x[53:66]),    # :,54:66
            np.mean(x[60:75])     # :,61:75
        ])
    elif x.ndim == 2:
        reduced_x = np.array([
            x[:,0],
            x[:,1],
            np.mean(x[:,2:4], axis=1),
            np.mean(x[:,3:5], axis=1),
            np.mean(x[:,4:7], axis=1),
            np.mean(x[:,5:9], axis=1),
            np.mean(x[:,7:11], axis=1),
            np.mean(x[:,9:13], axis=1),
            np.mean(x[:,11:16], axis=1),
            np.mean(x[:,15:20], axis=1),
            np.mean(x[:,19:25], axis=1),
            np.mean(x[:,24:32], axis=1),
            np.mean(x[:,31:40], axis=1),
            np.mean(x[:,39:49], axis=1),
            np.mean(x[:,48:59], axis=1),
            np.mean(x[:,53:66], axis=1),
            np.mean(x[:,60:75], axis=1)
        ])
    return reduced_x

def logspace_manual_20s(x,f):
    if x.ndim == 1:
        reduced_x = np.array([
            x[0],                # :,1
            np.mean(x[0:2]),     # :,1:2
            x[1],                # :,2
            np.mean(x[1:3]),     # :,2:3
            np.mean(x[2:4]),     # :,3:4
            np.mean(x[3:5]),     # :,4:5
            np.mean(x[4:7]),     # :,5:7
            np.mean(x[5:9]),     # :,6:9
            np.mean(x[7:11]),    # :,8:11
            np.mean(x[9:13]),    # :,10:13
            np.mean(x[11:16]),   # :,12:16
            np.mean(x[15:20])    # :,16:20
        ])
    elif x.ndim == 2:
        reduced_x = np.array([
            x[:,0],
            np.mean(x[:,0:2], axis=1),
            x[:,1],
            np.mean(x[:,1:3], axis=1),
            np.mean(x[:,2:4], axis=1),
            np.mean(x[:,3:5], axis=1),
            np.mean(x[:,4:7], axis=1),
            np.mean(x[:,5:9], axis=1),
            np.mean(x[:,7:11], axis=1),
            np.mean(x[:,9:13], axis=1),
            np.mean(x[:,11:16], axis=1),
            np.mean(x[:,15:20], axis=1)
        ])
    return reduced_x

def logspace_manual_10s(x,f):
    if x.ndim == 1:
        reduced_x = np.array([
            x[0],               # :,1
            np.mean(x[0:2]),    # :,1:2
            x[1],               # :,2
            np.mean(x[1:3]),    # :,2:3
            np.mean(x[2:4]),    # :,3:4
            np.mean(x[3:5]),    # :,4:5
            np.mean(x[4:7]),    # :,5:7
            np.mean(x[5:9]),    # :,6:9
            np.mean(x[7:10])    # :,8:10
        ])
    elif x.ndim == 2:
        reduced_x = np.array([
            x[:,0],
            np.mean(x[:,0:2], axis=1),
            x[:,1],
            np.mean(x[:,1:3], axis=1),
            np.mean(x[:,2:4], axis=1),
            np.mean(x[:,3:5], axis=1),
            np.mean(x[:,4:7], axis=1),
            np.mean(x[:,5:9], axis=1),
            np.mean(x[:,7:10], axis=1)
        ])
    return reduced_x