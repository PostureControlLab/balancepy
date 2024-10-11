import numpy as np
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as rfn
from scipy import fftpack as ft

def default_frf_opts():
    frf_opts = {
        'sr': 100, 
        'selFreq_start': 0,
        'selFreq_skip': 2,
        'selFreq_fmax': 2,
        'smoothPhase': True,
        'bootstrap_ci': False,
        }
    return frf_opts

def getSpec(x,sr=1000):
    # Inputs:
    #       x:      input time series (time along 2nd Dimension)
    #       sr:     sampling frequency of time series (pts/sec)
    #
    # Outputs:
    #       Sx:     amplitude spectrum; complex values
    #       Sxx:    power spectrum
    #       f:      frequencies (Hz)
    #
    # Sx is scaled such that the amplitude of a sine input is given by abs(Sx)
    # Sxx is scaled such that the integrated power density Sxx is equal to the 
    # mean power of the time domain input. sum(Sxx*df) = mean(data^2). 
    # df is the frequency bandwidth accounted for by each frequency point.
    #
    # (c) Lorenz Assländer, lorenz@asslaender.de, 12-Sept-2022
    
    N=np.size(x,0)              # number of samples in time axis
    f=np.arange(1,N/2+1) /N*sr  # frequency points for the output

    fk = ft.fft2(x)

    b= int(np.ceil(N/2)+1)
    y = fk[1:b,:]*2 # half sided spectrum

    Sx=1/N*y        # scaling to yield Sx, such that abs(Sx) = A

    Sxx = 1 / (sr*2*N) * abs(y)**2   # scaling to yield Sxx

    return Sx, Sxx, f

def getFRF(stim,resp,opts=0):
    
    if opts==0:
        opts = default_frf_opts()
        
    sr = opts['sr']
    yi,yii,f = getSpec(stim,sr)
    yo,yoo,_ = getSpec(resp,sr)

    # select Frequencies for output; convert options to range
    ind = np.where(f > opts['selFreq_fmax']) # find first index where f>fmax
    selFreq = range(opts['selFreq_start'],ind[0][0]+1,opts['selFreq_skip'])

    # calculate cross-power spectrum
    yoi = yo*np.conjugate(yi)
    yoi = 1/sr/2*np.size(stim,0) * yoi # scale cross spectrum by same factor as power spectra are scaled in getSpec
    
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

    if opts['smoothPhase']:
        Pha=smooth_phase(Pha,f)
        
    Coh=(abs(yoi_mean)**2) / (yii_mean*yoo_mean)


    t = np.arange(1,np.size(stim,0)+1) /sr
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

def fit_IC(x0):
    # from scipy import optimize as opt
    # cons = {'type':'eq', 'fun': IC_model}

    # opt.minimize(objective_function,x0,bounds=[],method='Nelder-Mead')
    # opt.basinhopping(objective_function,x0, niter=100)
    th = opts['IC_parameter']
    tf = IC_model(f,th)
    return tf

def objective_function(x):
    err = IC_model(x)
    return err

def IC_model(f,th=0):
    # input are frequency vector 'f' and parameter dict 'th'
    # f: float, th: float
    if th==0:
        opts = default_frf_opts()
        th = opts['IC_parameters']

    s = 1j*f*2*np.pi

    B = 1/(th['J']*s**2-th['mgh'])
    NC = th['Kp'] + th['Kd']*s
    TD = np.exp(-s*th['dt'])

    F = th['Glp']/s
    tf = (th['W']*NC*B*TD) / (1 - F*NC*TD + NC*B*TD)

    return tf
