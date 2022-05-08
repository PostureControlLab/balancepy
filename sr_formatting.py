from biomechanics import get_com


def get_sr_opts():
    sr_opts={
        'files': '', # 1 calls interface to select files; alternativ input is a filename or a list of filenames
        'filepath': '',
        'resample': 100, # numer gives desired sampling rate; 0 means no resampling
        'cut_cyc': True, # cut to cycles; True requires number of cycles to be defined in info file
        'discard_cyc': 1, # define cycles to be discarded from output
        'discard_time': [],
        'stim_name': 'stim_rz', # name of stimulus
        'resp_name': 'com_rz' # name of response
    }
    return sr_opts


def get_sr_data(opts=get_sr_opts()):
    # input options for files to be loaded
    #   1) files: filename - file must be in searchpath; files can be an array of filenames
    #   2) filepath (files empty): selects all readable *.csv files from specified folder
    #   3) files and filepath specified: filepath/file(s) are opened. filepath can be single folder or an array
    #   4) files and filepath empty: user interface is started which allows selection of folder or file(s) to be analyzed

    import pandas as pd

    # implement pre-processing
    # should contain check whether hip marker and shoulder marker were attached at the correct height
    # could also be implemented in recording...
    # if h_sm < h_hm    

    # handle different file input options
    if not opts['files'] and not opts['filepath']:
        print('UI file selection not implemented yet')
    elif not opts['files'] and opts['filepath']:
        print('selecting all files from filepath')
    elif opts['files']:
        # print('selecting all files using filepath if specified')
        # print('filepath:',opts['filepath'] + opts['files'])
        files = opts['filepath']+opts['files']



    # handle different response options
    if opts['resp_name'][0:3]=='com':
        df1 = pd.read_csv(files)
        df1 = get_com(df1,opts['resp_name'])
        
        df = df1[['time',opts['stim_name'],opts['resp_name']]]
    else:
        df = pd.read_csv(files, usecols=['time',opts['stim_name'],opts['resp_name']])

    
    
    # df = resample(df,opts['resample'])

    # if opts['cut_cyc']:
    #     cl = opts['cyc_length']
    #     print(cl)

    return df


def resample(df,sr):
    from scipy.interpolate import interp1d
    import pandas as pd
    import numpy as np

    ar = df.to_numpy() # convert dataframe to numpy

    t_end = max(df['time']) # get end of recording
    t_end = t_end - np.mod(t_end,1/sr) # cut at last resampled data point
    
    x=np.empty([round(t_end*sr),ar.shape[1]]) # preallocate new matrix

    newt = np.arange(1/sr, t_end+1/sr, 1/sr) # define time vector ]0, t_end]

    for c in range(len(ar[0])): # loop through columns and resamle
        x[:,c] = interp1d(ar[:,0], ar[:,c], kind='cubic')(newt)

    return pd.DataFrame(x, columns = df.columns) # output dataframe with resampled data


