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
P2 = PoissonGroup(3, 3*Hz)
P3 = PoissonGroup(3, 10*Hz)
G = NeuronGroup(5, izhi_model, threshold=threshold, reset=reset, method='euler')
G.v = c; G.u = b*c
S = Synapses(G, G, stdp_eq_e,on_pre=pre_eq, on_post=post_eq, method='euler')
S.connect(i = [0,0,1,1,2,2,3], j = [1,2,2,3,3,4,4])
S.w = [10,5,5,5,5,5,5]
S.delay = [3,15,7,18,6,15,4]*ms
S01 = Synapses(G,G,'w:1',on_pre='v+=w',method='euler')
S01.connect(i=0,j=1);S01.w = 10;S01.delay = 3*ms
S_PG = Synapses(P3, G, 'w:1 ',on_pre='v+=w',method='euler')
S_PG.connect(j='i+2')
S_PG.w = [10,10,10]
S_IG = Synapses(Inp,G, 'w:1 ',on_pre='v+=w',method='euler')
S_IG.connect(j='i')
S_IG.w = 20

w_mon = StateMonitor(S, 'w', record=True)
s_mon = SpikeMonitor(G)
v_mon = StateMonitor(G, 'v', record=True)
defaultclock.dt = 1*ms
run(n_iter*500*ms, report='text')



# %%
t1 = 0*500
t2 = 500*n_iter 
t_ind_s = np.where((s_mon.t/ms>t1) & (s_mon.t/ms < t2))[0]
plt.plot((s_mon.t/ms)[t_ind_s], s_mon.i[t_ind_s].T, '.')
plt.show()
# t_ind_w = np.where((w_mon13.t/ms>t1) & (w_mon13.t/ms<t2))[0]
# t_ind_s1 = np.where((s1_mon.t/ms>t1) & (s1_mon.t/ms<t2))[0]
# t_ind_s2 = np.where((s2_mon.t/ms>t1) & (s2_mon.t/ms<t2))[0]
# t_ind_s3 = np.where((s3_mon.t/ms>t1) & (s3_mon.t/ms<t2))[0]
# plt.plot((w_mon13.t/ms)[t_ind_w], (w_mon13.w[0])[t_ind_w],label='w13')
# plt.plot((w_mon23.t/ms)[t_ind_w], (w_mon23.w[0])[t_ind_w],label='w23')
# plt.legend()
# plt.show()
# plt.plot((s1_mon.t/ms)[t_ind_s1], (s1_mon.i)[t_ind_s1],'.',label='s1')
# plt.plot((s2_mon.t/ms)[t_ind_s2], (s2_mon.i)[t_ind_s2],'.',label='s2')
# plt.plot((s3_mon.t/ms)[t_ind_s3], (s3_mon.i)[t_ind_s3],'.',label='s3')
# plt.legend()
# plt.show()


# %%
plt.plot(w_mon.t/ms, w_mon.w[1],label='w02')
plt.plot(w_mon.t/ms, w_mon.w[2],label='w12')
plt.plot(w_mon.t/ms, w_mon.w[3],label='w13')
plt.plot(w_mon.t/ms, w_mon.w[4],label='w23')
plt.legend()
plt.show()
# %%

plt.plot((v_mon.t/ms)[720200:720300], (v_mon.v[3])[720200:720300],label='v2')
# %%
