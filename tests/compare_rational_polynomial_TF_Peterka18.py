import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import balancepy as bp
from balancepy.model_sim.peterka18 import Peterka18 as P18
import plotly.io as pio
pio.renderers.default = "browser"

## Purpose: Compare the frequency response of the polynomial transfer function including 
##          Pade approximation of the time delay with the exact formulation of the transfer 
##          function without Pade approximation. Thus to verify the correctness of the
##          polynomial transfer function implementation.

# Create a model instance of the balancepy Peterka 2018 model
height_m = 1.7
mass_kg = 80

model = P18(height_m=height_m, mass_kg=mass_kg)

# Obtain frequency response data of the polynomial formulation of the transfer function
f, mag, pha = model.bode()

# Create Bode plot

fig_bode = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07)

fig_bode.add_trace(go.Scatter(x=f, y=mag, mode='lines', name='Polynomial Model'), row=1, col=1)
fig_bode.add_trace(go.Scatter(x=f, y=pha, mode='lines', name='Polynomial Model'), row=2, col=1)

# Create the original model without pade approximation and not as polynomial
p = model.params.to_value_dict()
J = model.params['J'].value
mgh = model.params['mgh'].value
Kp = model.params['Kp'].value
Kd = model.params['Kd'].value
W = model.params['W'].value
dt = model.params['dt'].value
Kt = model.params['Kt'].value

s = 1j*f*2*np.pi

B = 1 / (p['J'] *s**2-p['mgh'])
NC = p['Kp'] + p['Kd'] *s
TD = np.exp(-s *p['dt'])

F = p['Kt'] /s
tf = (p['W'] *NC *B *TD)  / (1 - F *NC *TD + NC *B *TD)

# add traces to the plot
fig_bode.add_trace(go.Scatter(x=f, y=abs(tf), mode='lines', name='Original Model'), row=1, col=1)
fig_bode.add_trace(go.Scatter(x=f, y=bp.phase(tf,f), mode='lines', name='Original Model'), row=2, col=1)

# format figure and display
fig_bode.update_xaxes(type='log', row=1, col=1) 
fig_bode.update_xaxes(type='log', title_text='Frequency (Hz)', row=2, col=1)
fig_bode.update_yaxes(type='log', title_text='Magnitude', row=1, col=1)
fig_bode.update_yaxes(title_text='Phase (deg)', row=2, col=1)

fig_bode.update_layout(height=600, width=700, title_text="Bode Diagram")

fig_bode.show()
