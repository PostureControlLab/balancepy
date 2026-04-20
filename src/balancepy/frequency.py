from numpy.typing import NDArray
import numpy as np


def spectrum(
    data: NDArray[np.number],
    samplingrate_Hz: float
    ):
    """Calculate scaled amplitude and power spectra of a time domain signal.

        Amplitude spectrum is scaled such that the amplitude of a sine input 
        is given by abs(Sx)
        Power spectrum Sxx is scaled such that sum(Sxx*df) = mean(data^2), 
        where df is the frequency spacing.

    Parameters
    ----------
        data : array_like 
            time domain signal, can be 1D or 2D.
        samplingrate_Hz : float
            Sampling rate in Hz

    Returns
    -------
        Sx : array_like
            Scaled amplitude spectrum. 1D array for 1D input, 2D array for 2D input.
        Sxx : array_like
            Scaled power spectrum. 1D array for 1D input, 2D array for 2D input.
        freq : array_like
            Frequencies in Hz.
    """
    
    N=np.size(data,0)              # number of samples in time axis
    freq=np.arange(1,N/2+1) /N*samplingrate_Hz  # frequency points for the output

    fk = np.fft.fft(data, axis=0)

    b= int(np.ceil(N/2)+1)
    if fk.ndim == 1:
        y = fk[1:b] * 2  # half sided spectrum for 1D array
    elif fk.ndim == 2:
        y = fk[1:b, :] * 2  # half sided spectrum for 2D array

    Sx=1/N*y        # scaling to yield Sx, such that abs(Sx) = A

    Sxx = 1 / (samplingrate_Hz*2*N) * abs(y)**2   # scaling to yield Sxx

    return Sx, Sxx, freq

def coherence(yi,yo):
    """Calculate the coherence between two signals in the frequency domain.

    Parameters
    ----------
        yi : NDArray[np.number]
            Spectrum of multiple cycles of the stimulus.
        yo : NDArray[np.number]
            Spectrum of multiple cycles of the response.
        
    Returns
    -------
        coh : NDArray[np.number]
            Coherence between the two signals.
    """
    
    # calculate cross-power spectrum
    yoi = yo*np.conjugate(yi)

    yoi_mean=np.mean(yoi,1)
    yii_mean=np.mean(abs(yi)**2,1)
    yoo_mean=np.mean(abs(yo)**2,1)

    coh=(abs(yoi_mean)**2) / (yii_mean*yoo_mean)

    return coh

def frf(yi,yo):
    """Calculate the frequency response function (FRF).

    Parameters
    ----------
        yi : array_like
            Complex stimulus spectrum; can be multiple cycles.
        yo : array_like
            Complex response spectrum; can be multiple cycles.
        
    Returns
    -------
        frf : array_like
            Complex frequency response function (FRF).
    """
            
    # Calculate cross-power spectrum
    yoi = yo * np.conjugate(yi)
    yii = yi * np.conjugate(yi)

    
    if yoi.ndim == 1:
        H = yoi / yii
    else:
        yoi_mean = np.mean(yoi, 1)
        yii_mean = np.mean(yii, 1)

        H = yoi_mean / yii_mean

    return H


def phase(frf, f=None):
    """Calculate the phase of the frequency response function (FRF).

        Function performs additional phase unwrapping.

    Parameters
    ----------
        frf : array_like
            Complex frequency response function (FRF).
        f : array_like, optional
            Frequencies corresponding to the FRF. If provided, applies a reference
            polynomial for phase unwrapping. If None, returns raw angle.
    Returns
    -------
        pha : array_like
            Phase of the frequency response function (FRF) in degrees.
    """
    pha = np.angle(frf, deg=True)

    # If frequencies are provided, apply reference polynomial for phase unwrapping
    if f is not None:
        # create polynom roughly following a typical Phase curve of human sway responses + 180deg for modulo of 360deg
        # p_ref = 20-100*f-26*f**2 - 180
        p_ref = 20 - 180*f - 25*f**2 - 180
        pha = np.mod(pha - p_ref, 360) + p_ref
    
    return pha


