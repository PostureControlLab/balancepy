from math import floor
from biomechanics import get_com
import numpy as np
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as rfn

def get_sr_opts():
    sr_opts={
        'files': '', # 1 calls interface to select files; alternativ input is a filename or a list of filenames
        'filepath': '',
        'resample': 100, # numer gives desired sampling rate; 0 means no resampling
        't_end': 0, # desired sequence end for resampling; if 0, recording time is used
        'cut_cyc': True, # cut to cycles; True requires number of cycles to be defined in info file
        'discard_cyc': [0], # define cycles to be discarded from output
        'cyc_start': 0, # start sample of first cycle
        'cyc_length': 0, # is ignored if 0 and if ncyc is specified
        'ncyc': 1, # is ignored if 0
        'stim_name': 'stim_tz', # name of stimulus
        'resp_name': 'com_rap' # name of response
    }
    return sr_opts


def get_filenames(opts):
    # input options for files to be loaded
    #   1) files: filename - file must be in searchpath; files can be an array of filenames
    #   2) filepath (files empty): selects all readable *.csv files from specified folder
    #   3) files and filepath specified: filepath/file(s) are opened. filepath can be single folder or an array
    #   4) files and filepath empty: user interface is started which allows selection of folder or file(s) to be analyzed
    # handle different file input options
    if not opts['files'] and not opts['filepath']:
        print('UI file selection not implemented yet')
    elif not opts['files'] and opts['filepath']:
        print('selecting all files from filepath')
    elif opts['files']:
        # print('selecting all files using filepath if specified')
        # print('filepath:',opts['filepath'] + opts['files'])
        files = opts['filepath']+opts['files']

    return files

def get_raw_data(opts):
    fname = get_filenames(opts)

    df = np.genfromtxt(fname, delimiter=',', names=True)

    import biomechanics as bm
    df = bm.legacy_rename(df)

    time = df['time']

    stim = df[opts['stim_name']]

    resp_name = opts['resp_name']
    if resp_name[0:3]=='com':
        if resp_name=='com_tz' or resp_name=='com_tap':
                swdir = 'tz'
                angle = False
        elif resp_name=='com_rx' or resp_name=='com_rap':
                swdir = 'tz'
                angle = True
        elif resp_name=='com_tx' or resp_name=='com_tml':
                swdir = 'tx'
                angle = False
        elif resp_name=='com_rz' or resp_name=='com_rml':
                swdir = 'tx'
                angle = True
        else: 
                print('get_com input not recognized; should be com_tap/_rap/_tml/_rml')
                return

        resp = df['sho_'+swdir]
        resp2 = df['hip_'+swdir]

    else:
        resp = df[resp_name]
        resp2 = []

    return time, resp, stim, resp2

def get_sr_data(opts):
    # implement pre-processing
    # should contain check whether hip marker and shoulder marker were attached at the correct height
    # could also be implemented in recording...
    # if h_sm < h_hm    

    fname = get_filenames(opts)

    time = np.genfromtxt(fname, delimiter=',', names=True, usecols=['time'])
    time = rfn.structured_to_unstructured(time)

    # handle different response options
    if opts['resp_name'][0:3]=='com':
        resp = get_com(fname,opts['resp_name'])
    else:
        resp = np.genfromtxt(fname, delimiter=',', names=True, usecols=[opts['resp_name']])
        resp = rfn.structured_to_unstructured(resp)

    if 'stim_name' in opts:
        stim = np.genfromtxt(fname, delimiter=',', names=True, usecols=[opts['stim_name']])
        stim = rfn.structured_to_unstructured(stim)

    if 'resample' in opts and opts['resample'] != 0:
        time_rec = time
        time = resample(time,time_rec,opts)
        resp = resample(resp,time_rec,opts)
        if 'stim_name' in opts:
            stim = resample(stim,time_rec,opts)

    if 'cut_cyc' in opts and opts['cut_cyc']:
        resp = cut_to_cycles(resp,opts)
        time = cut_to_cycles(time,opts)
        # time = time[0:len(resp)]
        if 'stim_name' in opts:
            stim = cut_to_cycles(stim,opts)
        
    return time, resp, stim

def cut_to_cycles(data,opts):
    
    if 'cyc_length' in opts and opts['cyc_length'] != 0 and 'ncyc' in opts and opts['ncyc'] != 0:
        if data.size != opts['cyc_start'] + opts['cyc_length'] * opts['ncyc']:
            print('Warning: both cyc_length and ncyc are specified. cyc_length will be ignored')
    elif 'cyc_length' in opts and opts['cyc_length'] != 0:
       opts['ncyc'] = int( np.floor( (data.size - opts['cyc_start']) / opts['cyc_length']) )    
    elif 'ncyc' in opts and opts['ncyc'] != 0:
        opts['cyc_length'] = int( np.floor( (data.size - opts['cyc_start']) / opts['ncyc']) )

    data_out = np.empty( [ opts['cyc_length'], opts['ncyc'] ]) # preallocate new matrix

    a = np.arange(opts['ncyc'])
    
    for n in a:
        i_start = opts['cyc_start'] + n * opts['cyc_length']
        i_end = opts['cyc_start'] + (n+1) * opts['cyc_length']
        data_out[:,n] = data[i_start:i_end]

    # remove cycles that are marked in opts to be discarded
    ind = np.ones(len(a)) # create ncyc-long list of ones
    ind[opts['discard_cyc']] = 0 # set all discard cycles to zero in list

    data_out = data_out[:,ind==1] # select only cycles marked with 1

    return data_out
    


def resample(data,time_rec,opts):
    # time_rec are the recorded time stamps
    # new time stamps are derived from new sampling rate 'sr'
    # last point in resampled data is the last sampling point of the new sr within recorded time
    if 't_end' in opts and opts['t_end'] != 0:
        t_end = opts['t_end']
    else:
        t_end = max(time_rec) # get end of recording
        t_end = t_end - np.mod(t_end,1/sr) # cut at last resampled data point

    sr = opts['resample']
    newt = np.arange(1/sr, t_end+1/sr, 1/sr) # define time vector ]0, t_end]

    out = interp1d(time_rec[:,0], data[:,0], kind='cubic', fill_value='extrapolate')(newt)
    return out