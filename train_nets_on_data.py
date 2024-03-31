from numpy import *
import pandas as pd
import pickle
from matplotlib.pyplot import *
import scipy.io as sio
import sys
from modules.Network_fitting import *
from modules.RecurrentNets import *
import torch
import torch.nn as nn
from joblib import Parallel, delayed
import time
from scipy.linalg import schur
import modules.data_analysis_tools as dat
from modules.TrainingParams import *

phi = lambda x: nn.functional.softplus(x - 0, 2)

##
f = './data/PPC_data.pkl'
df = pd.read_pickle(f)
df = df.sort_values(by=['type'],ascending=False)
df = df.reset_index(drop=True)
signature = df.type.map({'pyramidal':1, 'unknown':1, 'non pyramidal':-1}).to_list()
signature = array(signature)
df = dat.normalize_data_by_peak_of_preferred_trial(df, columns = ['Ca_trial_mean_bR','Ca_trial_mean_wL'])

activity_right = np.stack(df.Ca_trial_mean_bR.values)[None]
activity_left = np.stack(df.Ca_trial_mean_wL.values)[None]
activity = np.concatenate((activity_left, activity_right), axis=0)
activity = transpose(activity, (0,2,1))

Nstimuli, T, N = activity.shape

E,I,E1,I1,E2,I2 = load('./data/neuron_idx.npy', allow_pickle=True) #indices of choice selective E/I neurons
NE1 = len(E1)
NI1 = len(I1)
NE2 = len(E2)
NI2 = len(I2)


##
# target activity has shape n_stimuli x n_time x n_cells
target_activity = activity

# external stimulus has shape n_stimuli x n_time x n_cells
sigma = 1
dt = 0.035
Ext_input = generate_Ext_input_2seq_onlyE(T, NE1, NI1, NE2, NI2, dt, sigma, tau=1., amplitude_pref = 1, amplitude_nonpref = 0.2)

##

model = Rajan_fitting(target_activity, Ext_input)
model.set_net_params(dt=0.033, tau=1, phi=phi, signature=signature)
model.train_net(n_epochs = 500, learning_rate = 0.02)

fit_A = model.state_net['fit_recurrent_connections']
fit_R = model.state_net['fit_activity']
fit_R0 = model.state_net['fit_initial_condition']

##

