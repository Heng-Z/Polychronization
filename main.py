#%%
import numpy as np

M = 100
D = 20
Ne = 800;Ni = 200;N = Ne+Ni
a = np.vstack((0.02*np.ones((Ne,1)),0.1*np.ones((Ni,1))))# parameter of Izhikevich neuron
d = np.vstack(8*(np.ones((Ne,1)),2*np.ones((Ni,1)))) # parameter of Izhikevich neuron
sm = 10

post = np.ceil(np.vstack((N*np.random.rand(Ne,M),Ne*np.random.rand(Ni,M)))) # post-syn neurons of nuerons 1~1000
s = np.vstack([6*np.ones((Ne,M)),-5*np.ones((Ni,M))])
sd = np.zeros((N,M))
pre = [np.where((post==i) & (s>0)) for i in range(Ne)] # pre excitatory neurons
# delay for each post neuron
delay = np.ones_like(post)

# initial value
v = -65*np.ones((N,1))
u = 0.5*v
STDP = np.zeros((N,1001+D))
firing = np.array([[N,1]])
# main run loop 
seconds = 100
for sec in range(seconds):
    for t in range(1000):
        I = np.zeros((N,1))
        # random thalamic input
        I[np.random.randint(N,size=1)] = 20
        # identify firing neurons
        fired = np.where(v>=30)[0]
        v[fired] = -65
        u[fired] = u[fired]+d[fired]
        STDP[fired,t+D] = 0.1
        # update sd for pre-syn neurons
        for k in range(len(fired)):
            # update the sd of 
            sd[pre[fired[k]]] = sd[pre[fired[k]]]+ STDP[pre[fired[k]][0],
                                                        t+D-np.ceil(pre[fired[k]][1]/(M/D))]
        # update firing table [ firing time, neuron number ]
        firing = np.r_[firing,np.c_[t*np.ones((len(fired),1)),fired.reshape(-1,1)]]
        k = firing.shape[0]
        while firing[k,0]>t-D-1:
            # update sd for pre-syn weights of neurons fired before
            delay = t-firing[k,0]
            if delay >=1:
                ind = post[firing[k][1],(delay-1)*M/D:delay*M/D] # indices of post-syn neurons with delay = 'delay'
                I[ind] = I[ind]+s[firing[k][1],(delay-1)*M/D:delay*M/D]
                sd[firing[k][1],(delay-1)*M/D:delay*M/D] = sd[firing[k][1],(delay-1)*M/D:delay*M/D] \
                    -1.2 * STDP[ind,t+D]
                k=k-1
        # update v and u
        v = v+0.5*(0.04*v**2+5*v+140-u+I)
        v = v+0.5*(0.04*v**2+5*v+140-u+I)
        u = u+a*(0.2*v-u)
        STDP[:,t+D+1] = 0.95*STDP[:,t+D]
    STDP[:,:D] = STDP[:,1000:1000+D]
    res_fire = np.where(firing[:,0]>1000-D)[0]
    firing = np.c_[firing[res_fire:,0]-1000,firing[res_fire:,1]]
    s[:Ne,:] = max(0,min(sm,s[:Ne,:]+sd[:Ne,:]+0.01))
    sd = 0.9*sd

            


# %%
