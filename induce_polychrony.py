#%%
from brian2 import *
# set_device('cpp_standalone', directory='STDP_standalone')
Ne = 800
Ni = 200
tau0 = 1*ms
taus = 10*ms
taupre = 20*ms
taupost = taupre
defaultclock.dt = 0.5*ms
F = 1*Hz
# a = 0.02/ms
b = 0.2
c=-65
# d = 8
Iin = 10
wmax = 10
dApre = 0.2
dApost = -dApre * taupre / taupost * 1.2
# Difine Neuron Groups
izhi_model = '''dv/dt = (0.04*v**2+5*v+140-u)/tau0 : 1
                    du/dt = a*(b*v-u) : 1
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
n_indices = array([0,1,2,3,4,5,6,7])
s_times = array([100,106,104,104,101,103,107,103])*ms
Inp_G = SpikeGeneratorGroup(8, n_indices, s_times)
Pe = PoissonGroup(Ne, F)
Pi = PoissonGroup(Ni, F)
# Define Synaptice connection: EE, EI, IE, II, PE,PI
# Firstly, Define STDP eqns
# stdp_eq = '''w : 1
#                 dApre/dt = -Apre / taupre : 1 (event-driven)
#                 dApost/dt = -Apost / taupost : 1 (event-driven)
#                 wtot_pre = w : 1 (summed)'''
# pre_eq = '''v += w
#                     Apre = clip(Apre+ dApre,0,2*dApre)
#                     w = clip(w + Apost, 0, wmax)'''
# post_eq = '''Apost = clip(Apost+ dApost,2*dApost,0)
#                      w = clip(w + Apre, 0, wmax)'''

stdp_eq = '''   dw/dt = s/tau0 + 2*clip(wmax-w,-10,0)/tau0 + 2*clip(-w,0,10)/tau0 : 1 (clock-driven)
                ds/dt = -s / taus : 1 (clock-driven)
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)'''
pre_eq = '''v += w
                    Apre+=dApre
                    s = clip(Apost,2*dApost,0)'''
post_eq = '''Apost += dApost
                     s = clip(Apre,0,2*dApre)'''
# Define Synapses and set parameters
S_EE = Synapses(Ge,Ge,stdp_eq,on_pre=pre_eq,on_post=post_eq)
S_EE.connect(condition='i!=j', p=0.1)
S_EE.w = 6
S_EE.delay = 'ceil(rand()*20)*ms'
S_EI = Synapses(Ge,Gi,stdp_eq,on_pre=pre_eq,on_post=post_eq)
S_EI.connect(condition='i!=j', p=0.1)
S_EI.w = 6
S_EI.delay = 'ceil(rand()*20)*ms'
S_IE = Synapses(Gi,Ge,'w:1',on_pre='v-=w')
S_IE.connect(condition='i!=j', p=0.1)
S_IE.w = 5
S_IE.delay = 'ceil(rand()*20)*ms'
S_II = Synapses(Gi,Gi,'w:1',on_pre='v-=w')
S_II.connect(condition='i!=j', p=0.1)
S_II.w = 5
S_II.delay = 'ceil(rand()*20)*ms'
S_PE = Synapses(Pe,Ge,'w:1',on_pre='v+=w')
S_PE.connect(i='j')
S_PE.w = 10
S_PI = Synapses(Pi,Gi,'w:1',on_pre='v+=w')
S_PI.connect(i='j')
S_PI.w = 10
S_Inp_E = Synapses(Inp_G,Ge,'w:1',on_pre='v+=w')
S_Inp_E.connect(j='100*i')
S_Inp_E.w = 20
# Define Monitors
w_E_mon = StateMonitor(S_EE, 'w', record=np.arange(10))
w_I_mon = StateMonitor(S_IE, 'w', record=np.arange(10))
# s_E_mon = StateMonitor(S_EE, 's', record=np.arange(10))
Apre_mon = StateMonitor(S_EE, 'Apre', record=np.arange(10))
Ge_spike = SpikeMonitor(Ge)
Gi_spike = SpikeMonitor(Gi)
Pe_spike = SpikeMonitor(Pe)
Pi_spike = SpikeMonitor(Pi)
#%%
# Run the simulation
run_per_iter = 500*ms
run_iter = 80
n_indices_all = [*n_indices]*run_iter
s_times_all = [t + i*run_per_iter for i in range(run_iter) for t in s_times]
Inp_G.set_spikes(n_indices_all, s_times_all)
run(run_per_iter*run_iter, report='text')
# %%
t_indices = np.where((Ge_spike.t/ms <16000) & (Ge_spike.t/ms > 14700))[0]
plot((Ge_spike.t/ms)[t_indices], Ge_spike.i[t_indices], '.')
#%%
plot(w_E_mon.w.T[31300:31500])
# %%
Ge_spike_time = Ge_spike.t/ms
Ge_spike_id = Ge_spike.i
Gi_spike_time = Gi_spike.t/ms
Gi_spike_id = Gi_spike.i
# save spike time and id as npz file
np.savez('spike_induce.npz',Ge_spike_time=Ge_spike_time,Ge_spike_id=Ge_spike_id,Gi_spike_time=Gi_spike_time,Gi_spike_id=Gi_spike_id)

# %%
