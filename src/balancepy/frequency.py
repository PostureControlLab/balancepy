from numpy.typing import NDArray
import numpy.lib.recfunctions as rfn
import numpy as np
import skrf as rf
import balancepy as bp

def frequency_analysis(
    xi: NDArray[np.number],
    xo: NDArray[np.number],
    samplingrate_Hz: int,
    selected_frequencies_index = None,
    selected_frequencies_type = None,
    bootstrap_samples: int = 0
    ) -> NDArray:
    """calculates frequency response functions (FRFs).

    Args:
        stim (NDArray[np.number]): 2D stimulus sequence with cycles in rows
        resp (NDArray[np.number]): 2D response data with cycles in rows
        samplingrate (float): sampling rate in samples/second
        selected_frequencies_type: 'all' or 'prts' or array of indices
        smoothing: None or function performing smoothing of the FRF

    Returns:
        NDArray[np.number]: matrix with frequency domain outputs
        f: frequency
        frf: frequency response function
        gain: gain of frequency response function
        pha: phase of frequency response function
        coh: coherence
    """

    yi,yii,f = spectrum(xi,samplingrate_Hz)
    yo,yoo,_ = spectrum(xo,samplingrate_Hz)
    
    if selected_frequencies_index is not None:
        f   = f[selected_frequencies_index]
        yi  = yi[selected_frequencies_index,:]
        yo  = yo[selected_frequencies_index,:]
    elif selected_frequencies_index is None and selected_frequencies_type is not None:
        selected_frequencies_index = get_frequency_selection(selected_frequencies_type, samplingrate_Hz, f)
        f   = f[selected_frequencies_index]
        yi  = yi[selected_frequencies_index,:]
        yo  = yo[selected_frequencies_index,:]

    # mean spectra
    yi_mean = abs(np.mean(yi,1))
    yo_mean = abs(np.mean(yo,1))
        
    # Calculate FRF, Magnitude and Phase of FRF, as well as Coherence
    # FRF from position data - Pintelon & Schoukens eq 2-17
    frf = bp.frf(yi, yo)
    coh = coherence(yi,yo)
    
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
        data (NDArray[np.number]): 1D or 2D data array
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
    if fk.ndim == 1:
        y = fk[1:b] * 2  # half sided spectrum for 1D array
    elif fk.ndim == 2:
        y = fk[1:b, :] * 2  # half sided spectrum for 2D array

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


