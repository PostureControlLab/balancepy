from curses.ascii import isblank
import matplotlib.pyplot as plt
import numpy as np
from src.sr_formatting import resample
from biomechanics import get_com
from src.sr_formatting import get_sr_data
from src.sr_formatting import get_sr_opts
from src.sr_formatting import cut_to_cycles
from importlib import reload
from src.pc_frequency import IC_model, getFRF, getSpec


fname = '/Users/macbookpro/Nextcloud2/isorropia/data/test_s1_t2_vrdata.csv'

opts = get_sr_opts()
opts['files'] = fname
# opts['filepath'] = '/Users/macbookpro/Nextcloud2/isorropia/data/'

print(opts)
time, resp, stim = get_sr_data(opts)

yi, yii, f = getSpec(stim,100)
yo, yoo,_ = getSpec(resp,100)

FD, TD = getFRF(stim,resp)

tf = IC_model(f)

plt.subplot(3,2,1)
plt.plot(TD['t'],TD['xi_mean'])
plt.subplot(3,2,2)
plt.plot(TD['t'],TD['xo_mean'])

plt.subplot(3,2,3)
plt.plot(FD['f'],abs(FD['yi_mean']))
plt.subplot(3,2,4)
plt.plot(FD['f'],abs(FD['yo_mean']))

plt.subplot(3,2,5)
plt.semilogx(FD['f'],FD['Gain'])
plt.subplot(3,2,6)
plt.semilogx(FD['f'],FD['Pha'])


plt.show()