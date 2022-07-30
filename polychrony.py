#%%
import numpy as np
import time
class Polychain:
    def __init__(self):
        self.Ne = 800 # number of excitatory neurons
        self.Ni = 200 # number of inhibitory neurons
        self.N = self.Ne+self.Ni # total number of neurons
        self.M = 100 # number of synapses per neuron
        self.wmax = 20
        self.we = 6
        self.wi = -5
        self.D_max = 20 # maximum delay
        self.D_min = 1 # minimum delay
        self.a = np.vstack((0.02*np.ones((self.Ne,1)),0.1*np.ones((self.Ni,1)))) # parameter of Izhikevich neuron
        self.d = np.vstack((8*(np.ones((self.Ne,1)),2*np.ones((self.Ni,1))))) # parameter of Izhikevich neuron
        self.b = 0.2 # parameter of Izhikevich neuron
        self.c = -65 #mv parameter of Izhikevich neuron
        self.dt = 1 #ms
        self.tau = 20 #ms time constant of STDP
        self.STDP_A =0.1 # Potentiation and depression amplitude of STDP
        # self.STDP_Ap = 0.1 # Potentiation amplitude
        # self.STDP_An = 0.1 # Depression amplitude
        self.get_initial_value()

    def get_initial_value(self):
        self.W = np.zeros((self.N,self.N)) # synaptic weight
        self.D = np.zeros((self.N,self.N)) # synaptic delay 
        for i in range(self.Ne):
            output_ind = np.random.choice(self.N,self.M,replace=False)
            self.W[output_ind,i] = self.we
            self.D[output_ind,i] = np.random.randint(self.D_min,self.D_max,self.M)
        for i in range(self.Ne,self.N):
            output_ind = np.random.choice(self.N,self.M,replace=False)
            self.W[output_ind,i] = self.wi
            self.D[output_ind,i] = np.random.randint(self.D_min,self.D_max,self.M)
        self.Mask = (self.W != 0).astype(np.int8) # mask of synaptic weight
        self.u = np.zeros((self.N,1)) # restore
        self.v = -65*np.ones((self.N,1)) # membrane potential
        self.S = np.zeros((self.N,self.N)) # Spike traveling time
        self.A = np.zeros((self.N,self.N)) # STDP auxiliary variable
        self.B = np.zeros((self.N,self.N)) # STDP auxiliary variable
        self.dW = np.zeros((self.N,self.N)) # derivative of synaptic weight

    def neuron_cable_update(self,I):
        t1 =time.time()
        self.v = self.v + self.dt*(0.04*self.v**2+5*self.v+140-self.u+I)
        t15 = time.time()
        self.u = self.a*(self.b*self.v-self.u)*self.dt+self.u
        t2 = time.time()
        fire_ind = np.where(self.v>=30)[0]
        self.v[fire_ind] = self.c
        self.u[fire_ind] = self.u[fire_ind]+self.d[fire_ind]
        # update the traveling time of spike on the cables
        on_road_ind = np.where(self.S>0)
        self.S[on_road_ind] = self.S[on_road_ind]+1
        self.S[:,fire_ind] = self.Mask[:,fire_ind]
        t3 = time.time()
        print('update v time: ',t15-t1)
        print('update u time: ',t2-t15)
        print('spike mat update time: ',t3-t2)
        return fire_ind

    def STDP_update(self,fire_ind,arrive_ind):
        # update the STDP -- Exponential decay with constant tau
        self.A = self.A*(1 - self.dt/self.tau)
        self.B = self.B*(1 - self.dt/self.tau)
        self.A[self.A<1e-3] = 0
        self.B[self.B<1e-3] = 0
        # update pre and post synapse spike record
        self.A[fire_ind,:] = 1*self.Mask[fire_ind,:] # post synaptic spike onset record
        self.B[arrive_ind] = 1 # pre synaptic spike arrive record
        # update the weight and weight derivative according to the STDP rule
        C = self.A * self.B * self.STDP_A
        stdp_ind = np.where(C!=0) # where stdp 
        self.dW = self.dW*0.9
        self.dW = self.dW + (np.sign(self.A - self.B)*C)
        self.W = self.W + self.dW # A>B means pre-synaptic spike is later than post-synaptic spike, then potentiate it
        self.W[:,:self.Ne] = np.clip(self.W[:,:self.Ne],0,self.wmax)
        self.W[:,self.Ne:] = np.clip(self.W[:,self.Ne:],-self.wmax,0)
        self.A[stdp_ind] = 0
        self.B[stdp_ind] = 0

#%%
pc = Polychain()
T = 1000*100 # simulation time
dt = pc.dt
t = np.arange(0,T,dt)
fire_ind = None
for ti in range(len(t)):
    t1 = time.time()
    # Index of synapse that spikes arrive
    arrive_mat = ((pc.D - pc.S == 0) & pc.Mask == 1).astype(np.int8) # with 1s represent the synapse that spikes arrive
    arrive_ind = np.where(arrive_mat==1)
    t2 = time.time()
    print('find arrive spike time: ',t2-t1)
    pc.S[arrive_ind] = 0
    # Calculate the input current
    t3 = time.time()
    I = np.sum(arrive_mat*pc.W,axis=1)
    fire_ind = pc.neuron_cable_update(I)
    t4 = time.time()
    print('update neuron state time: ',t4-t3)
    # Update weight and STDP auxiliary variable
    pc.STDP_update(fire_ind, arrive_ind)
    t5 = time.time()
    print('update stdp time: ',t5-t4)
    if ti ==0:
        break


# %%
