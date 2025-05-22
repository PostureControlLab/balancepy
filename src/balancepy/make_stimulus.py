import numpy as np
import warnings


def make_stimulus(type='mseq', ncyc=1, ampl=1, vel=0, samplingrate=1000, t_state=0.25, power=4, base=3, seq=1, shift=0):
    """
    Generate a stimulus based on m-sequence or other types.

    Args:
        type (str): Type of stimulus ('Peterka2002', 'prts80', 'mseq', etc.).
        ncyc (int): Number of cycles to repeat the stimulus.
        ampl (float): Amplitude of the stimulus.
        ampl2 (float): Secondary amplitude (not used in this implementation).
        vel (float): Velocity scaling factor.
        samprate (int): Sampling rate in Hz.
        t_state (float): Time per state in seconds.
        power (int): Power for m-sequence generation.
        base (int): Base for m-sequence generation.
        seq (int): Sequence instantiation to use.
        shift (int): Cyclical shift of the sequence.

    Returns:
        np.ndarray: Generated stimulus.
    """
    if type == 'Peterka2002':
        ms = mseq(3, 5, shift=37, which_seq=2)  # Peterka 2002 stimulus
    elif type == 'prts_20s':
        ms = mseq(3, 4, shift=59, which_seq=1)  # Standard 20s stimulus
    elif type == 'mseq':
        ms = mseq(base, power, shift=shift, which_seq=seq)
    elif type == 'dual':
        raise NotImplementedError("Dual type is not implemented.")
    elif type == 'sine':
        raise NotImplementedError("Sine type is not implemented.")
    else:
        raise ValueError(f"Unknown type: {type}")

    # if a velocity is specified, scale the stimulus to the velocity
    if vel != 0:
        ms = ms * vel

    # integrate the m-sequence to get the position sequence of the stimulus
    if not (t_state * samplingrate).is_integer():
        warnings.warn("samplingrate does not allow duration of individual prts states")
    stim = np.cumsum(np.repeat(ms, int(t_state * samplingrate)) / samplingrate)

    # 
    if vel == 0:
        stim = stim / (np.max(stim) - np.min(stim)) * ampl

    stim = np.tile(stim, ncyc)
    return stim


def mseq(base_val, power_val, shift=1, which_seq=1):
    """
    Generate a maximum length sequence (m-sequence).

    Args:
        base_val (int): Number of sequence levels (2, 3, or 5 allowed).
        power_val (int): Power, so that sequence length is base_val^power_val - 1.
        shift (int): Cyclical shift of the sequence.
        which_seq (int): Sequence instantiation to use.

    Returns:
        np.ndarray: Generated maximum length sequence.

    Adapted from the MATLAB code by Giedrius T. Buracas, SNL-B, Salk Institute.
    Reference:
    - Davies, W.D.T. (1970). *System Identification for Self-Adaptive Control*. Wiley-Interscience.
    - Buracas, G.T. & Boynton, G.M. (2002). Efficient Design of Event-Related fMRI Experiments Using M-sequences. NeuroImage, 16, 801–813.
    """

    if base_val not in [2, 3, 5]:
        raise ValueError("base_val must be 2, 3, or 5.")

    bit_num = base_val**power_val - 1
    register = np.ones(power_val, dtype=int)

    # Define taps for base_val = 2
    taps = {
        (2, 2): [[1, 2]],
        (2, 3): [[1, 3], [2, 3]],
        (2, 4): [[1, 4], [3, 4]],
        (2, 5): [[2, 5], [3, 5], [1, 2, 3, 5], [2, 3, 4, 5], [1, 2, 4, 5], [1, 3, 4, 5]],
        (2, 6): [[1, 6], [5, 6], [1, 2, 5, 6], [1, 4, 5, 6], [1, 3, 4, 6], [2, 3, 5, 6]],
        (2, 7): [[1, 7], [6, 7], [3, 7], [4, 7], [1, 2, 3, 7], [4, 5, 6, 7], [1, 2, 5, 7], [2, 5, 6, 7],
                 [2, 3, 4, 7], [3, 4, 5, 7], [1, 3, 5, 7], [2, 4, 6, 7], [1, 3, 6, 7], [1, 4, 6, 7],
                 [2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 7], [1, 2, 4, 5, 6, 7], [1, 2, 3, 5, 6, 7]],
        (2, 8): [[1, 2, 7, 8], [1, 6, 7, 8], [1, 3, 5, 8], [3, 5, 7, 8], [2, 3, 4, 8], [4, 5, 6, 8],
                 [2, 3, 5, 8], [3, 5, 6, 8], [2, 3, 6, 8], [2, 5, 6, 8], [2, 3, 7, 8], [1, 5, 6, 8],
                 [1, 2, 3, 4, 6, 8], [2, 4, 5, 6, 7, 8], [1, 2, 3, 6, 7, 8], [1, 2, 5, 6, 7, 8]],
        (2, 9): [[4, 9], [5, 9], [3, 4, 6, 9], [3, 5, 6, 9], [4, 5, 8, 9], [1, 4, 5, 9], [1, 4, 8, 9], [1, 5, 8, 9],
            [2, 3, 5, 9], [4, 6, 7, 9], [5, 6, 8, 9], [1, 3, 4, 9], [2, 7, 8, 9], [1, 2, 7, 9], [2, 4, 7, 9],
            [2, 5, 7, 9], [2, 4, 8, 9], [1, 5, 7, 9], [1, 2, 4, 5, 6, 9], [3, 4, 5, 7, 8, 9], [1, 3, 4, 6, 7, 9],
            [2, 3, 5, 6, 8, 9], [3, 5, 6, 7, 8, 9], [1, 2, 3, 4, 6, 9], [1, 5, 6, 7, 8, 9], [1, 2, 3, 4, 8, 9],
            [1, 2, 3, 7, 8, 9], [1, 2, 6, 7, 8, 9], [1, 3, 5, 6, 8, 9], [1, 3, 4, 6, 8, 9], [1, 2, 3, 5, 6, 9],
            [3, 4, 6, 7, 8, 9], [2, 3, 6, 7, 8, 9], [1, 2, 3, 6, 7, 9], [1, 4, 5, 6, 8, 9], [1, 3, 4, 5, 8, 9],
            [1, 3, 6, 7, 8, 9], [1, 2, 3, 6, 8, 9], [2, 3, 4, 5, 6, 9], [3, 4, 5, 6, 7, 9], [2, 4, 6, 7, 8, 9],
            [1, 2, 3, 5, 7, 9], [2, 3, 4, 5, 7, 9], [2, 4, 5, 6, 7, 9], [1, 2, 4, 5, 7, 9], [2, 4, 5, 6, 7, 9],
            [1, 3, 4, 5, 6, 7, 8, 9], [1,2,3,4,5,6,8,9]],
        (2, 10): [[3, 10], [7, 10], [2, 3, 8, 10], [2, 7, 8, 10], [1, 3, 4, 10], [6, 7, 9, 10], [1, 5, 8, 10],
            [2, 5, 9, 10], [4, 5, 8, 10], [2, 5, 6, 10], [1, 4, 9, 10], [1, 6, 9, 10], [3, 4, 8, 10],
            [2, 6, 7, 10], [2, 3, 5, 10], [5, 7, 8, 10], [1, 2, 5, 10], [5, 8, 9, 10], [2, 4, 9, 10],
            [1, 6, 8, 10], [3, 7, 9, 10], [1, 3, 7, 10], [1, 2, 3, 5, 6, 10], [4, 5, 7, 8, 9, 10],
            [2, 3, 6, 8, 9, 10], [1, 2, 4, 7, 8, 10], [1, 5, 6, 8, 9, 10], [1, 2, 4, 5, 9, 10],
            [2, 5, 6, 7, 8, 10], [2, 3, 4, 5, 8, 10], [2, 4, 6, 8, 9, 10], [1, 2, 4, 6, 8, 10],
            [1, 2, 3, 7, 8, 10], [2, 3, 7, 8, 9, 10], [3, 4, 5, 8, 9, 10], [1, 2, 5, 6, 7, 10],
            [1, 4, 6, 7, 9, 10], [1, 3, 4, 6, 9, 10], [1, 2, 6, 8, 9, 10], [1, 2, 4, 8, 9, 10],
            [1, 4, 7, 8, 9, 10], [1, 2, 3, 6, 9, 10], [1, 2, 6, 7, 8, 10], [2, 3, 4, 8, 9, 10],
            [1, 2, 4, 6, 7, 10], [3,4,6,8,9,10], [2,4,5,7,9,10], [1,3,5,6,8,10], [3,4,5,6,9,10],
 		    [1,4,5,6,7,10], [1,3,4,5,6,7,8,10], [2,3,4,5,6,7,9,10], [3,4,5,6,7,8,9,10], [1,2,3,4,5,6,7,10],
		    [1,2,3,4,5,6,9,10], [1,4,5,6,7,8,9,10], [2,3,4,5,6,8,9,10], [1,2,4,5,6,7,8,10], [1,2,3,4,6,7,9,10], [1,3,4,6,7,8,9,10]],
        (2, 11): [[9, 11]],
        (2, 12): [[6, 8, 11, 12]],
        (2, 13): [[9, 10, 12, 13]],
        (2, 14): [[4, 8, 13, 14]],
        (2, 15): [[14, 15]],
        (2, 16): [[4, 13, 15, 16]],
        (2, 17): [[14, 17]],
        (2, 18): [[11, 18]],
        (2, 19): [[14, 17, 18, 19]],
        (2, 20): [[17, 20]],
        (2, 21): [[19, 21]],
        (2, 22): [[21, 22]],
        (2, 23): [[18, 23]],
        (2, 24): [[17, 22, 23, 24]],
        (2, 25): [[22, 25]],
        (2, 26): [[20, 24, 25, 26]],
        (2, 27): [[22, 25, 26, 27]],
        (2, 28): [[25, 28]],
        (2, 29): [[27, 29]],
        (2, 30): [[7, 28, 29, 30]],

        # Define taps for base_val = 3
        (3, 2): [[2, 1], [1, 1]],
        (3, 3): [[0, 1, 2], [1, 0, 2], [1, 2, 2], [2, 1, 2]],
        (3, 4): [[0, 0, 2, 1], [0, 0, 1, 1], [2, 0, 0, 1], [2, 2, 1, 1], 
                 [2, 1, 1, 1], [1, 0, 0, 1], [1, 2, 2, 1], [1, 1, 2, 1]],
        (3, 5): [[0, 0, 0, 1, 2], [0, 0, 1, 2, 2], [0, 2, 0, 2, 2], [0, 2, 1, 0, 2], 
                 [0, 2, 1, 1, 2], [0, 1, 2, 0, 2], [0, 1, 1, 2, 2], [2, 0, 0, 1, 2], 
                 [2, 0, 2, 0, 2], [2, 0, 2, 2, 2], [2, 2, 0, 2, 2], [2, 2, 2, 1, 2], 
                 [2, 2, 1, 2, 2], [2, 1, 2, 2, 2], [2, 1, 1, 0, 2], [1, 0, 0, 0, 2], 
                 [1, 0, 0, 2, 2], [1, 0, 1, 1, 2], [1, 2, 2, 2, 2], [1, 1, 0, 1, 2], [1, 1, 2, 0, 2]],
        (3, 6): [[0, 0, 0, 0, 2, 1], [0, 0, 0, 0, 1, 1], [0, 0, 2, 0, 2, 1], [0, 0, 1, 0, 1, 1], 
                 [0, 2, 0, 1, 2, 1], [0, 2, 0, 1, 1, 1], [0, 2, 2, 0, 1, 1], [0, 2, 2, 2, 1, 1], 
                 [2, 1, 1, 1, 0, 1], [1, 0, 0, 0, 0, 1], [1, 0, 2, 1, 0, 1], [1, 0, 1, 0, 0, 1], 
                 [1, 0, 1, 2, 1, 1], [1, 0, 1, 1, 1, 1], [1,2,0,2,2,1], [1,2,0,1,0,1], [1,2,2,1,2,1],
                 [1,2,1,0,1,1], [1,2,1,2,1,1], [1,2,1,1,2,1], [1,1,2,1,0,1], [1,1,1,0,1,1], [1,1,1,2,0,1], [1,1,1,1,1,1]],
        (3, 7): [[0, 0, 0, 0, 2, 1, 2], [0, 0, 0, 0, 1, 0, 2], [0, 0, 0, 2, 0, 2, 2], [0, 0, 0, 2, 2, 2, 2],
                 [0, 0, 0, 2, 1, 0, 2], [0, 0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1, 2], [0, 0, 2, 2, 2, 0, 2],
                 [0, 0, 2, 2, 1, 2, 2], [0, 0, 2, 1, 0, 0, 2], [0, 0, 2, 1, 2, 2, 2], [0, 0, 1, 0, 2, 1, 2],
                 [0, 0, 1, 0, 1, 1, 2], [0, 0, 1, 1, 0, 1, 2], [0, 0, 1, 1, 2, 0, 2], [0, 2, 0, 0, 0, 2, 2],
                 [0, 2, 0, 0, 1, 0, 2], [0, 2, 0, 0, 1, 1, 2], [0, 2, 0, 2, 2, 0, 2], [0, 2, 0, 2, 1, 2, 2],
                 [0, 2, 0, 1, 1, 0, 2], [0, 2, 2, 0, 2, 0, 2], [0, 2, 2, 0, 1, 2, 2], [0, 2, 2, 2, 2, 1, 2],
                 [0, 2, 2, 2, 1, 0, 2], [0, 2, 2, 1, 0, 1, 2], [0, 2, 2, 1, 2, 2, 2]],

        # Define taps for base_val = 5
        (5, 2): [[4, 3], [3, 2], [2, 2], [1, 3]],
        (5, 3): [[0, 2, 3], [4, 1, 2], [3, 0, 2], [3, 4, 2], [3, 3, 3], [3, 3, 2], [3, 1, 3], [2, 0, 3],
                [2, 4, 3], [2, 3, 3], [2, 3, 2], [2, 1, 2], [1, 0, 2], [1, 4, 3], [1, 1, 3]],
        (5, 4): [[0, 4, 3, 3], [0, 4, 3, 2], [0, 4, 2, 3], [0, 4, 2, 2], [0, 1, 4, 3], [0, 1, 4, 2], [0, 1, 1, 3],
                [0, 1, 1, 2], [4, 0, 4, 2], [4, 0, 3, 2], [4, 0, 2, 3], [4, 0, 1, 3], [4, 4, 4, 2], [4, 3, 0, 3],
                [4, 3, 4, 3], [4, 2, 0, 2], [4, 2, 1, 3], [4, 1, 1, 2], [3, 0, 4, 2], [3, 0, 3, 3], [3, 0, 2, 2],
                [3, 0, 1, 3], [3, 4, 3, 2], [3, 3, 0, 2], [3, 3, 3, 3], [3, 2, 0, 3], [3, 2, 2, 3], [3, 1, 2, 2],
                [2, 0, 4, 3], [2, 0, 3, 2], [2, 0, 2, 3], [2, 0, 1, 2], [2, 4, 2, 2], [2, 3, 0, 2], [2, 3, 2, 3],
                [2, 2, 0, 3], [2, 2, 3, 3], [2, 1, 3, 2], [1,0,4,3], [1,0,3,3], [1,0,2,2], [1,0,1,2], [1,4,1,2],
                [1,3,0,3], [1,3,1,3], [1,2,0,2], [1,2,4,3], [1,1,4,2]]                 
    }

    # Get the taps for the given base_val and power_val
    if (base_val, power_val) in taps:
        tap_list = taps[(base_val, power_val)]
    else:
        raise ValueError(f"M-sequence {base_val}^{power_val} is not defined.")

    if which_seq < 1 or which_seq > len(tap_list):
        which_seq = (which_seq - 1) % len(tap_list) + 1

    weights = np.zeros(power_val, dtype=int)
    if base_val == 2:
        weights[np.array(tap_list[which_seq - 1]) - 1] = 1
    else:
        weights = np.array(tap_list[which_seq - 1])

    ms = np.zeros(bit_num, dtype=int)
    for i in range(bit_num):
        ms[i] = np.sum(weights * register) % base_val
        register = np.roll(register, 1)
        register[0] = ms[i]

    if shift:
        shift = shift % len(ms)
        ms = np.concatenate((ms[shift:], ms[:shift]))

    if base_val == 2:
        ms = ms * 2 - 1
    elif base_val == 3:
        ms[ms == 2] = -1
    elif base_val == 5:
        ms[ms == 4] = -1
        ms[ms == 3] = -2
    else:
        raise ValueError("Invalid base_val!")

    return ms