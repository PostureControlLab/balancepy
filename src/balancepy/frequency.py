from numpy.typing import NDArray
import numpy.lib.recfunctions as rfn
import numpy as np
import skrf as rf
import balancepy as bp


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


