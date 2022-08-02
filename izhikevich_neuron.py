#%%
import numpy as np
import matplotlib.pyplot as plt

class IzhiNeuron:
    def __init__(self):
        self.a = 0.02 # 0.02 for excitatory and 0.1 for inhibitory 
        self.b = 0.2
        self.c = -65
        self.d = 8 # 8 for excitatory and 2 for inhibitory
        self.dt = 1
        self.u = 0
        self.v = -65
    def update(self,I):
        self.v = self.v + self.dt/2*(0.04*self.v**2+5*self.v+140-self.u+I)
        self.u = self.a*(self.b*self.v-self.u)*self.dt+self.u
        self.v = self.v + self.dt/2*(0.04*self.v**2+5*self.v+140-self.u+I)
        if self.v > 30:
            self.v = self.c
            self.u = self.u + self.d
#%%
neuron = IzhiNeuron() 
neuron.a = 0.1 # for inhibitory
neuron.d = 2
dt = neuron.dt
t = np.arange(0,300,dt)
ut = np.zeros(t.shape)
vt = np.zeros(t.shape)
# Get and ramp up and ramp down shape signal in numpy
ramp_sig = np.zeros_like(t)
ramp_sig[0:int(len(t)/2)] = t[0:int(len(t)/2)]/500*20
ramp_sig[int(len(t)/2):] = 40-t[int(len(t)/2):]/500*20
# Spike train
spike_train = np.zeros_like(t)
spike_train[250] = 10
spike_train[251] = 10 
#
input_sig = spike_train # ramp_sig or spike_train
for ti in range(len(t)):
    neuron.update(input_sig[ti])
    ut[ti] = neuron.u
    vt[ti] = neuron.v
#%%
plt.plot(t,vt)
plt.plot(t,input_sig)
plt.show()
# %%
