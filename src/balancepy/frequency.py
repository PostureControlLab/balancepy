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


def frequency_response_function(
    stim: NDArray[np.number],
    resp: NDArray[np.number],
    sampling_rate: float,
    selFreq_start: int=1,
    selFreq_skip: int=2,
    selFreq_fmax: float=2,
    smoothPhase: bool=True,
    
) -> NDArray:
    """calculates frequency response functions (FRFs).

    Args:
        stim (NDArray[np.number]): 2D stimulus sequence with cycles in rows
        resp (NDArray[np.number]): 2D response data with cycles in rows
        sr (float): sampling rate in samples/second
        selFreq_start (int, optional): start frequency point of output FRF (1 is base-frequency of 1 cycle). Defaults to 1.
        selFreq_fmax (int, optional): end frequency in Hz for FRF calculation. Defaults to 2.
        selFreq_skip (int, optional): number of frequencies to be skipped in output FRF. Defaults to 2 (skips every second frequency).

    Returns:
        NDArray[np.number]: matrix with frequency domain outputs
        NDArray[np.number]: matrix with time domain outputs
    """
            
    yi,yii,f = spectrum(stim,sampling_rate)
    yo,yoo,_ = spectrum(resp,sampling_rate)

    # select Frequencies for output; convert options to range
    ind = np.where(f > selFreq_fmax) # find first index where f>fmax
    selFreq = range(selFreq_start,ind[0][0]+1,selFreq_skip)

    # calculate cross-power spectrum
    yoi = yo*np.conjugate(yi)
    yoi = 1/sampling_rate/2*np.size(stim,0) * yoi # scale cross spectrum by same factor as power spectra are scaled in getSpec
    
    # reduce to selected frequencies
    f   = f[selFreq]
    yi  = yi[selFreq,:]
    yo  = yo[selFreq,:]
    yii = yii[selFreq,:]
    yoo = yoo[selFreq,:]
    yoi = yoi[selFreq,:]
       
    # mean spectra
    yi_mean=np.mean(yi,1)
    yo_mean=np.mean(yo,1)
    
    yoi_mean=np.mean(yoi,1)
    yii_mean=np.mean(yii,1)
    yoo_mean=np.mean(yoo,1)
        
    # Calculate FRF, Magnitude and Phase of FRF, as well as Coherence
    # FRF from position data - Pintelon & Schoukens eq 2-17
    FRF=yo_mean / yi_mean
    Gain=abs(FRF)
    Pha=np.angle(FRF,deg=True)

    if smoothPhase:
        Pha=smooth_phase(Pha,f)
        
    Coh=(abs(yoi_mean)**2) / (yii_mean*yoo_mean)


    t = np.arange(1,np.size(stim,0)+1) /sampling_rate
    xi_mean = np.mean(stim,1)
    xo_mean = np.mean(resp,1)


    FD = rfn.merge_arrays([
                np.array(f,    dtype=[('f','<f8')]),
                np.array(yi_mean, dtype=[('yi_mean','complex')]),
                np.array(yo_mean, dtype=[('yo_mean','complex')]),
                np.array(FRF, dtype=[('FRF','complex')]),
                np.array(Gain, dtype=[('Gain','<f8')]),
                np.array(Pha,  dtype=[('Pha','<f8')]),
                np.array(Coh,  dtype=[('Coh','<f8')])
                ],
                flatten = True, usemask = False)

    TD = rfn.merge_arrays([
                np.array(t,  dtype=[('t','<f8')]),
                np.array(xi_mean,  dtype=[('xi_mean','<f8')]),
                np.array(xo_mean,  dtype=[('xo_mean','<f8')]),
                ],
                flatten = True, usemask = False)

    return FD, TD

def smooth_phase(pha,f):
    # create polynom roughly following a typical Phase curve + 180deg for modulo of 360deg
    p_ref = 100-500*f+100*f**2 - 180
    pha = np.mod(pha-p_ref,360) + p_ref
    return pha





