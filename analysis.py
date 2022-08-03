#%%
import numpy as np
import matplotlib.pyplot as plt
# spike_data = np.load('spike_time_id.npz')
spike_data = np.load('spike_induce.npz')
#%%
Ge_spike_time = spike_data['Ge_spike_time']
Ge_spike_id = spike_data['Ge_spike_id']
Gi_spike_time = spike_data['Gi_spike_time']
Gi_spike_id = spike_data['Gi_spike_id']
#%%
t = 9000*1000
t_indices = np.where((Ge_spike_time > t+100) &  (Ge_spike_time < t+250))
plt.plot(Ge_spike_time[t_indices], Ge_spike_id[t_indices], '.b')
# plt.plot(Gi_spike_time[:10000], Gi_spike_id[:10000]+800, '.g')
plt.show()
# %%
