#%%
from brian2 import *
import matplotlib.pyplot as plt
# set_device('cpp_standalone', directory='STDP_standalone')
M =1 
tau0 = 1*ms
taus = 10*ms
taupre = 20*ms
taupost = taupre
defaultclock.dt = 0.5*ms
# F = 5*Hz
a = 0.02/ms
b = 0.2
c=-65
d = 8
wmax = 10
wtotmax = 20
dApre = 1
dApost = -dApre * taupre / taupost * 1.1
n_iter = 1500
# neuron and synapse model
izhi_model = '''dv/dt = (0.04*v**2+5*v+140-u)/tau0 : 1
                    du/dt = a*(b*v-u) : 1
                    wetot : 1
                    witot : 1
                    a : 1/second
                    d : 1'''
threshold = 'v>30'
reset = 'v=c; u+=d' 

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

s_time = [200+500*i for i in range(n_iter)]*ms
Inp = SpikeGeneratorGroup(1, [0]*n_iter,s_time)
P2 = PoissonGroup(1, 7*Hz)
P3 = PoissonGroup(1, 5*Hz)
G1 = NeuronGroup(1, izhi_model, threshold=threshold, reset=reset, method='euler'); G1.v = c; G1.u = b*c
G2 = NeuronGroup(1, izhi_model, threshold=threshold, reset=reset, method='euler'); G2.v = c; G2.u = b*c
G3 = NeuronGroup(1, izhi_model, threshold=threshold, reset=reset, method='euler'); G3.v = c; G3.u = b*c
S1 = Synapses(Inp,G1,'w:1',on_pre='v+=w',method='euler')
S1.connect()
S1.w = 20
S2 = Synapses(P2,G3,'w:1',on_pre='v+=w',method='euler')
S2.connect()
S2.w = 20
S3 = Synapses(P3,G3,'w:1',on_pre='v+=w',method='euler')
S3.connect()
S3.w = 10
S12 = Synapses(G1,G2,'w:1',on_pre='v+=w',method='euler')
S12.connect()
S12.w = 20
S12.delay = 5*ms
S13 = Synapses(G1,G3,stdp_eq_e,on_pre=pre_eq,on_post=post_eq,method='euler')
S13.connect()
S13.w = 5
S13.delay = 15*ms
S23 = Synapses(G2,G3,stdp_eq_e,on_pre=pre_eq,on_post=post_eq,method='euler')
S23.connect()
S23.w = 5
S23.delay = 5*ms

w_mon13 = StateMonitor(S13,'w',record=True)
w_mon23 = StateMonitor(S23,'w',record=True)
v1_mon = StateMonitor(G1,'v',record=True)
v3_mon = StateMonitor(G3,'v',record=True)
s0_mon = SpikeMonitor(Inp)
s1_mon = SpikeMonitor(G1)
s2_mon = SpikeMonitor(G2)
s3_mon = SpikeMonitor(G3)
defaultclock.dt = 1*ms
run(n_iter*500*ms, report='text')



# %%
t1 = 0*500
t2 = t1+500*n_iter 
t_ind_w = np.where((w_mon13.t/ms>t1) & (w_mon13.t/ms<t2))[0]
t_ind_s1 = np.where((s1_mon.t/ms>t1) & (s1_mon.t/ms<t2))[0]
t_ind_s2 = np.where((s2_mon.t/ms>t1) & (s2_mon.t/ms<t2))[0]
t_ind_s3 = np.where((s3_mon.t/ms>t1) & (s3_mon.t/ms<t2))[0]
plt.plot((w_mon13.t/ms)[t_ind_w], (w_mon13.w[0])[t_ind_w],label='w13')
plt.plot((w_mon23.t/ms)[t_ind_w], (w_mon23.w[0])[t_ind_w],label='w23')
plt.legend()
plt.show()
plt.plot((s1_mon.t/ms)[t_ind_s1], (s1_mon.i)[t_ind_s1],'.',label='s1')
plt.plot((s2_mon.t/ms)[t_ind_s2], (s2_mon.i)[t_ind_s2],'.',label='s2')
plt.plot((s3_mon.t/ms)[t_ind_s3], (s3_mon.i)[t_ind_s3],'.',label='s3')
plt.legend()
plt.show()


# %%
t1 = 000
t2 = 2000
t_ind_v = np.where((v3_mon.t/ms>t1) & (v3_mon.t/ms<t2))[0]
plt.plot((v3_mon.t/ms)[t_ind_v], v3_mon.v[0][t_ind_v],label='v1')
plt.legend()
plt.show()
# %%
