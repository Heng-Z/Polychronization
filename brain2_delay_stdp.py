#%%
from brian2 import *
# set_device('cpp_standalone', directory='STDP_standalone')
tau0 = 1*ms
taupre = 20*ms
taupost = taupre
F = 200*Hz
a = 0.02/ms
b = 0.2
c=-65
d = 8
Iin = 10
gmax = 20
dApre = 2
dApost = -dApre * taupre / taupost * 1.2


G1 = NeuronGroup(1, '''dv/dt = (0.04*v**2+5*v+140-u)/tau0 : 1
                    du/dt = a*(b*v-u) : 1''',
                    threshold='v>30', reset='v=c; u+=d', method='euler')
G1.v = c
G1.u = b*c

G2 = NeuronGroup(1, '''dv/dt = (0.04*v**2+5*v+140-u)/tau0 : 1
                    du/dt = a*(b*v-u) : 1''',
                    threshold='v>30', reset='v=c; u+=d', method='euler')
                
P1 = PoissonGroup(1, F)
P2 = PoissonGroup(1, F)
stdp_eq = '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)'''
pre_eq = '''v += w
                    Apre += dApre
                    w = clip(w + Apost, 0, gmax)'''
post_eq = '''Apost += dApost
                     w = clip(w + Apre, 0, gmax)'''
S1 = Synapses(P1,G1,on_pre='v+=Iin')
S1.connect()
S2 = Synapses(P1,G2,on_pre='v+=Iin')
S2.connect()
S2.delay = 2*ms
S3 = Synapses(G1,G2,stdp_eq,on_pre=pre_eq,on_post=post_eq)
S3.connect()
S3.w = 10
mon = StateMonitor(S3, 'w', record=[0])
v1_mon = StateMonitor(G1, 'v', record=[0])
p1_mon = SpikeMonitor(P1)
s1_mon = SpikeMonitor(G1)
s2_mon = SpikeMonitor(G2)
defaultclock.dt = 1*ms
run(1*second, report='text')
# %%

# %%
plot((s1_mon.t/ms), s1_mon.i,'.')
plot((v1_mon.t/ms),v1_mon.v[0])
# %%
plot((p1_mon.t/ms), p1_mon.i,'.')
plot((v1_mon.t/ms),v1_mon.v[0])
# %%
plot(mon.t/ms,mon.w[0])
plot(s1_mon.t/ms,s1_mon.i,'o')#pre
plot(s2_mon.t/ms,s2_mon.i,'v')#post

# %%
