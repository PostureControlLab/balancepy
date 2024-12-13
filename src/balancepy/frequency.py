from numpy.typing import NDArray
import numpy.lib.recfunctions as rfn
import numpy as np
import scipy.fftpack as ft

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

    fk = ft.fft2(data)

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

def frequency_analysis(
    stim: NDArray[np.number],
    resp: NDArray[np.number],
    sampling_rate: float,
    selected_frequencies: NDArray[np.int32] = 0,
    smoothPhase: bool=True,
    average_across_freq: bool=True
    
) -> NDArray:
    """calculates frequency response functions (FRFs).

    Args:
        stim (NDArray[np.number]): 2D stimulus sequence with cycles in rows
        resp (NDArray[np.number]): 2D response data with cycles in rows
        sr (float): sampling rate in samples/second
        selected_frequencies (NDArray[np.int32], optional): 1D frequencies as multiples of base freq. Defaults to range(1,2,1).
        smoothPhase (bool, optional): smooth phase curve. Defaults to True.
        average_across_freq (bool, optional): average across frequencies. Defaults to True.

    Returns:
        NDArray[np.number]: matrix with frequency domain outputs
        f: frequency
        yi: stimulus amplitude spectrum
        yo: response amplitude spectrum
        frf: frequency response function
        gain: gain of frequency response function
        pha: phase of frequency response function
        coh: coherence
        NDArray[np.number]: matrix with time domain outputs
        t: time
        xi: stimulus averaged across cylces
        xo: response averaged across cylces
    """
            
    if selected_frequencies == 0:
        selected_frequencies = range(1, 2 * np.size(resp, 0) / sampling_rate, 2)

    yi,yii,f = spectrum(stim,sampling_rate)
    yo,yoo,_ = spectrum(resp,sampling_rate)
    
    # reduce to selected frequencies
    f   = f[selected_frequencies]
    yi  = yi[selected_frequencies,:]
    yo  = yo[selected_frequencies,:]

    # mean spectra
    yi_mean=np.mean(yi,1)
    yo_mean=np.mean(yo,1)
            
    # Calculate FRF, Magnitude and Phase of FRF, as well as Coherence
    # FRF from position data - Pintelon & Schoukens eq 2-17
    frf=yo_mean / yi_mean
    stim_spec = abs(yi_mean)
    resp_spec = abs(yo_mean)
    coh = coherence(yi,yo)

    if average_across_freq:
        f = f_avg(f)
        frf = f_avg(frf)
        stim_spec = f_avg(stim_spec)
        resp_spec = f_avg(resp_spec)
        coh = f_avg(coh)

    gain=abs(frf)
    pha=np.angle(frf,deg=True)

    if smoothPhase:
        pha=smooth_phase(pha,f)

    FD = rfn.merge_arrays([
                np.array(f,    dtype=[('f','<f8')]),
                np.array(stim_spec, dtype=[('stim_spec','<f8')]),
                np.array(resp_spec, dtype=[('resp_spec','<f8')]),
                np.array(frf, dtype=[('frf','complex')]),
                np.array(gain, dtype=[('gain','<f8')]),
                np.array(pha,  dtype=[('phase','<f8')]),
                np.array(coh,  dtype=[('coherence','<f8')])
                ],
                flatten = True, usemask = False)

    t = np.arange(1,np.size(stim,0)+1) /sampling_rate
    xi_mean = np.mean(stim,1)
    xo_mean = np.mean(resp,1)

    TD = rfn.merge_arrays([
                np.array(t,  dtype=[('time','<f8')]),
                np.array(xi_mean,  dtype=[('stim','<f8')]),
                np.array(xo_mean,  dtype=[('resp','<f8')]),
                ],
                flatten = True, usemask = False)

    return FD, TD

def smooth_phase(pha,f):
    # create polynom roughly following a typical Phase curve of human sway responses + 180deg for modulo of 360deg
    p_ref = 100-500*f+100*f**2 - 180
    pha = np.mod(pha-p_ref,360) + p_ref
    return pha

def f_avg(x):
    if x.ndim == 1:
            reduced_x = np.array([
                x[0],
                np.mean(x[0:2]),
                x[1],
                np.mean(x[1:3]),
                np.mean(x[2:4]),
                np.mean(x[3:5]),
                np.mean(x[4:7]),
                np.mean(x[5:9]),
                np.mean(x[7:11]),
                np.mean(x[9:13]),
                np.mean(x[11:16]),
                np.mean(x[15:20])
            ])
    elif x.ndim == 2:
        reduced_x = np.array([
            x[0, :],
            np.mean(x[0:2, :], axis=0),
            x[1, :],
            np.mean(x[1:3, :], axis=0),
            np.mean(x[2:4, :], axis=0),
            np.mean(x[3:5, :], axis=0),
            np.mean(x[4:7, :], axis=0),
            np.mean(x[5:9, :], axis=0),
            np.mean(x[7:11, :], axis=0),
            np.mean(x[9:13, :], axis=0),
            np.mean(x[11:16, :], axis=0),
            np.mean(x[15:20, :], axis=0)
        ])

    return reduced_x



