import numpy as np
import pandas as pd
import pickle
from matplotlib.pyplot import *
import scipy.io as sio
import sys
sys.path.insert(0, './modules/')
from modules import utilities
from modules import data_analysis_tools as dat
from training_nets_tools import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import joblib
import time

from TrainingParams import *
device = torch.device('cpu')
n_jobs = joblib.cpu_count()

theta = 1
beta = 0.5
which_nl = sys.argv[1]
relu = lambda x,theta: nn.functional.softplus(x - theta, beta)
satu = lambda x,theta: 2 / (1 + torch.exp(-4.*(x-theta)))
dict_nl = dict(satu=satu, relu=relu)
nonlinearity = lambda x: dict_nl[which_nl](x,theta)

dict_g = {'g01':0.1, 'g02':0.2, 'g05':0.5, 'g08':0.8}
g = dict_g[sys.argv[2]]

f = './data/trial_avg_activity_n131.pkl'
# f = './data/trial_avg_activity.pkl'

df = pd.read_pickle(f)
df = df.sort_values(by=['type'],ascending=False)
df = df.reset_index(drop=True)
signature = df.type.map({'pyramidal':1, 'unknown':1, 'non pyramidal':-1}).to_list()
df = dat.normalize_data_by_peak_of_preferred_trial(df, columns = ['Ca_trial_mean_bR','Ca_trial_mean_wL'])

activity_right = np.stack(df.Ca_trial_mean_bR.values)[None]
activity_left = np.stack(df.Ca_trial_mean_wL.values)[None]
activity = np.concatenate((activity_left, activity_right), axis=0)

# idx_L = df[df.selectivity_MI=='Left'].index
# idx_R = df[df.selectivity_MI=='Right'].index
# idx_NS = df[(df.selectivity_MI=='Non') | (df.selectivity_MI=='Mixed')].index
# idx_c = [idx_L, idx_R, idx_NS]

idx_L = df[df.select_idx_MI < 0].index
idx_R = df[df.select_idx_MI > 0].index
idx_c = [idx_L, idx_R]

N = activity.shape[1]
seq_length = activity.shape[2]
t_max_target = (seq_length - 1) * dt_target
target_t = np.arange(0, t_max_target+dt_target, dt_target)
n_t_target = len(target_t)

dt_rec = np.diff(target_t)[0]
assert np.allclose(np.diff(target_t), dt_rec)
dt = dt_rec / rec_step

t_max = target_t[-1] + dt_rec
t = np.arange(0, t_max, dt)
n_t = len(t)

# input = set_input(n_t, N, dt, signature, idx_c, on_which='EI')

###---------------Sparsity values -------------
psparse = np.array([1])
#----------------------------------------------

time0 = time.time()
data_vs_par = pd.DataFrame({'psparse':[], 'data':[], 'num_trainings':[],'signature':[], 
                            'theta':[], 'idx_c':[],'dt':[], 'target':[]})

for i_par in range(len(psparse)):
    print(f'p={psparse[i_par]}', flush=True)

    def train_network_many_times():
        #---------------------
        input = set_input(n_t, N, dt, signature, idx_c, on_which='EI')
        #---------------------

        train = TrainNetOnTraces(activity, target_t, input, device)
        train.set_net_params(N, noise_std, dt, tau, g, signature, nonlinearity, psparse[i_par])
        train.train_net(rec_step=rec_step, n_epochs=n_epochs, verbose=False)
        return train.state_net

    data_training     = Parallel(n_jobs=n_jobs)(delayed(train_network_many_times)() for i in range(num_trainings))
    data_training     = pd.Series(data_training)
    data_training_all = pd.Series(dict(data_training=data_training))
    data_vs_par = data_vs_par.append({'psparse':psparse[i_par], 
                                      'data':data_training_all, 
                                      'num_trainings':num_trainings,
                                      'signature':signature, 
                                      'theta':theta, 
                                      'idx_c':idx_c,
                                      'dt':dt, 
                                      'target':activity
                                      }, ignore_index=True)

print(f'Time elapsed = {time.time()-time0}')

# filename = './results/data_training_' + which_nl + '_thfixed1_vs_psparse_'+ sys.argv[2] +'.pkl'

# filename = './results/data_training_' + which_nl + '_thfixed1_vs_psparse_'+ sys.argv[2] +'_192nets_beta05_inputEI.pkl'

filename = './results/data_training_131_inputEI_1_pt3.pkl'

data_vs_par.to_pickle(filename)



