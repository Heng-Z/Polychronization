#%%
import numpy as np
from brian2 import *

Ne = 800
Ni = 200
M = 100
tau0 = 1*ms
taus = 10*ms
taupre = 20*ms
taupost = taupre
defaultclock.dt = 0.5*ms
F = 20*Hz
# a = 0.02/ms
b = 0.2
c=-65
# d = 8
wmax = 10
wtotmax = M*1
dApre = 1
dApost = -dApre * taupre / taupost * 1.1
# Difine Neuron Groups
izhi_model = '''dv/dt = (0.04*v**2+5*v+140-u)/tau0 : 1
                    du/dt = a*(b*v-u) : 1
                    wetot : 1
                    witot : 1
                    a : 1/second
                    d : 1'''
threshold = 'v>30'
reset = 'v=c; u+=d'   
Ge = NeuronGroup(Ne, izhi_model, threshold=threshold, reset=reset, method='euler')
Ge.a = 0.02/ms
Ge.d = 8
Ge.v = c
Ge.u = b*c
Gi = NeuronGroup(Ni, izhi_model, threshold=threshold, reset=reset, method='euler')
Gi.a = 0.1/ms
Gi.d = 2
Gi.v = c
Gi.u = b*c
# Define Input Spikes
# np.random.seed(123)
n_indices =np.arange(20)
s_times = np.random.randint(101,121,20)*ms
Inp_G = SpikeGeneratorGroup(20, n_indices, s_times)
Pe = PoissonGroup(Ne, F)
Pi = PoissonGroup(Ni, F)
# Define Synaptice connection: EE, EI, IE, II, PE,PI
# Firstly, Define STDP eqns
stdp_eq_e = ''' dw/dt = - clip(wetot_pre + witot_pre - wtotmax,0,inf)/M/taus : 1 (clock-driven)
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)
                wetot_pre = w : 1 (summed)'''
stdp_eq_i = ''' dw/dt = - clip(wetot_pre + witot_pre - wtotmax,0,inf)/M/taus : 1 (clock-driven)
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)
                witot_pre = w : 1 (summed)'''
pre_eq = '''v += w
                    Apre = clip(Apre+ dApre,0,2*dApre)
                    w = clip(w + Apost, 0, wmax)'''
post_eq = '''Apost = clip(Apost+ dApost,2*dApost,0)
                     w = clip(w + Apre, 0, wmax)'''

# Define Synapses and set parameters
S_EE = Synapses(Ge,Ge,stdp_eq_e,on_pre=pre_eq,on_post=post_eq,method='euler')
S_EE.connect(p=0.1)
S_EE.w = 6
S_EE.delay = 'ceil(rand()*20)*ms'
S_EI = Synapses(Ge,Gi,stdp_eq_i,on_pre=pre_eq,on_post=post_eq,method='euler')
S_EI.connect(p=0.1)
S_EI.w = 6
S_EI.delay = 'ceil(rand()*20)*ms'
S_IE = Synapses(Gi,Ge,'w:1',on_pre='v-=w')
S_IE.connect(p=0.1)
S_IE.w = 5
S_IE.delay = 'ceil(rand()*20)*ms'
S_II = Synapses(Gi,Gi,'w:1',on_pre='v-=w')
S_II.connect(p=0.1)
S_II.w = 5
S_II.delay = 'ceil(rand()*20)*ms'
S_Inp_E = Synapses(Inp_G,Ge,'w:1',on_pre='v+=w')
S_Inp_E.connect(j='40*i')
S_Inp_E.w = 20
#%%
EE_w = np.zeros((Ne,Ne))
EI_w = np.zeros((Ne,Ni))
EE_w[S_EE.i[:],S_EE.j[:]] = S_EE.w[:]
EI_w[S_EI.i[:],S_EI.j[:]] = S_EI.w[:]
E_w = np.concatenate((EE_w,EI_w),axis=1)
EE_delay = np.zeros((Ne,Ne))
EI_delay = np.zeros((Ne,Ni))
EE_delay[S_EE.i[:],S_EE.j[:]] = S_EE.delay[:]/ms
EI_delay[S_EI.i[:],S_EI.j[:]] = S_EI.delay[:]/ms
E_delay = np.concatenate((EE_delay,EI_delay),axis=1)
inp_id = np.arange(20)
triplets = []
for i in inp_id:
    for j in range(i-1):
        d_s = (s_times[i] - s_times[j])/ms
        overlap_neuron = np.where((E_w[i*40,:]!=0) & (E_w[j*40,:]!=0))[0]
        for k in overlap_neuron:
            if abs(d_s - E_delay[j*40,k] + E_delay[i*40,k]) <=1+1e-6:
                print('potential member:',i*40,j*40,k)
                triplets.append((i*40,j*40,k))
# %%
