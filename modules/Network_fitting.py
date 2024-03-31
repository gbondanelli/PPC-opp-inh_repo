import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from numpy import random
from matplotlib.pyplot import *
# sys.path.insert(0, '/')
# from modules import data_analysis_tools as dat
from scipy.stats import pearsonr

class RNN(nn.Module):
    def __init__(self, Nstimuli, T, N, dt, tau, phi=None, signature=None):
        super(RNN, self).__init__()
        self.N = N
        self.T = T
        self.Nstimuli = Nstimuli
        self.signature = signature
        self.phi = phi
        self.dt = dt
        self.tau = tau
        if self.phi is None:
            self.phi = lambda x:x

        if self.signature is not None:
            mask_EI = np.ones(self.N)[:,None] @ self.signature[None,:]
            self.mask_EI = torch.tensor(mask_EI, dtype=torch.float32)
            self.mask = self.mask_EI

        self.mask_sparse = np.random.choice([1,0], p=[.2,.8], size=(self.N,self.N))
        self.mask_sparse = torch.tensor(self.mask_sparse, dtype=torch.float32)

        self.A = nn.Parameter(torch.Tensor(self.N, self.N))
        with torch.no_grad():
            self.A.normal_(std=0.2/np.sqrt(self.N))

        self.h0 = nn.Parameter(torch.Tensor(self.Nstimuli, self.N))
        with torch.no_grad():
            self.h0.normal_(std=0.2 / np.sqrt(self.N))

    def effective_W(self, W):
        if self.signature is not None:
            self.W_eff = torch.abs(W) * self.mask
        else:
            self.W_eff = W
        return self.W_eff

    def forward(self, input, h0=None, W=None, noise_std=0.0):
        # R0 is Nstimuli x N
        # input is Nstimuli x time x N

        rates = torch.zeros(self.Nstimuli, self.T, self.N)
        if W is None:
            J = self.effective_W(self.A)
        else:
            J = W

        dt  = self.dt
        tau = self.tau
        N = input.shape[2]

        for i_stim in range(self.Nstimuli):
            if h0 is None:
                h = self.h0[i_stim,:]
            else:
                h = h0[i_stim,:]
            rates[i_stim, 0, :] = self.phi(self.h0[i_stim,:])
            for i in range(self.T-1):
                rec_input = self.phi(h).matmul(J.t()) + input[i_stim, i, :]
                h = ((1 - dt/tau)*h
                     + dt/tau*rec_input
                     + np.sqrt(dt)/tau * noise_std * torch.randn(N))
                rates[i_stim, i+1, :] = self.phi(h)
        return rates

class Rajan_fitting(RNN):
    def __init__(self, target_activity, input):
        self.target_activity = target_activity
        self.input = input
        self.converging = True
        self.Nstimuli, self.T, self.N = target_activity.shape
        self.is_trained = False

    def set_net_params(self, dt, tau, phi=None, signature=None):
        Nstimuli, T, N = self.Nstimuli, self.T, self.N
        super(Rajan_fitting, self).__init__(Nstimuli, T, N, dt, tau, phi, signature)
        net = RNN(Nstimuli, T, N, dt, tau, phi, signature)
        self.net = net

    def train_net(self, n_epochs = 200, learning_rate = 0.01):
        loss_criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        self.losses = np.zeros(n_epochs)

        target_activity_torch = torch.from_numpy( self.target_activity.astype(np.float32) )
        input_torch = torch.from_numpy( self.input.astype(np.float32) )
        R0_torch = target_activity_torch[:,0,:]

        for i_epoch in range(n_epochs):
            optimizer.zero_grad()

            A_in = get_band_matrix(self.net.A.detach().numpy(), k=20, which='in')
            A_out = get_band_matrix(self.net.A.detach().numpy(), k=20, which='out')
            A_in = A_in.ravel()
            A_out = A_out.ravel()
            A_in = np.mean(A_in[A_in!=0])
            A_out = np.mean(A_out[A_out!=0])

            Y = self.net(input_torch)
            loss = loss_criterion(Y, target_activity_torch) + 10 * (my_relu(A_out,0.05) +  my_relu(-A_in,0.05))
            loss.backward()
            optimizer.step()

            loss.detach_()
            self.losses[i_epoch] = loss.item()
            print(f'Epoch={i_epoch}, Loss = {loss.item()}')

            # early stopping:
            if i_epoch > 150:
                last_losses = self.losses[i_epoch - 100: i_epoch]
                if np.mean(last_losses) > 50:
                    self.converging = False
                    break

        with torch.no_grad():
            fit_R = self.net(input_torch)
            fit_R = fit_R.detach().numpy()

        fit_A = self.net.A.detach().numpy().copy()
        if self.net.signature is not None:
            fit_A = abs(fit_A) * self.net.mask.numpy()

        fit_h0 = self.net.h0.detach().numpy().copy()
        fit_R0 = self.net.phi(self.net.h0).detach().numpy().copy()

        self.fit_activity = fit_R
        self.fit_recurrent_connections = fit_A
        self.fit_h0 = fit_h0
        self.fit_R0 = fit_R0
        self.state_net = dict(fit_recurrent_connections = self.fit_recurrent_connections  if self.converging else np.nan,
                              fit_activity = self.fit_activity if self.converging else np.nan,
                              fit_initial_condition = self.fit_R0 if self.converging else np.nan,
                              losses = self.losses if self.converging else np.nan)

        self.is_trained = True

    def regen_dynamics(self, input_this, h0, W, noise_std=0):
        with torch.no_grad():
            input_torch = torch.from_numpy(input_this.astype(np.float32))
            h0 = torch.from_numpy(h0.astype(np.float32))
            W = torch.from_numpy(W.astype(np.float32))
            rates = self.net(input_torch, h0, W, noise_std)
            return rates.numpy().copy()

def generate_Ext_input(T, N, dt, sigma, tau=1., amplitude=1.):
    X = np.zeros((T,N))
    # X[0] = np.zeros(N) # initial condition of external input
    X[0] = np.random.randn(N)
    # X[0] = np.ones(N) # this was the one used in Kuan et al (but it does not make much difference)
    for t in range(T-1):
        X[t+1] = X[t] - dt/tau * X[t] + sigma*np.sqrt(dt)*np.random.randn(N)
    return X*amplitude

def generate_Ext_input_2seq(T, N, dt, sigma, tau=1., amplitude_pref = 1, amplitude_nonpref = 0.5):
    # amplitude_pref/nonpref modulate the amplitude of stim specifi ext input to neuron preferring/non-preferring a given stimulus
    Ext_input_stim1_pool1 = generate_Ext_input(T, N//2, dt, sigma, tau, amplitude_pref)
    Ext_input_stim2_pool1 = generate_Ext_input(T, N//2, dt, sigma, tau, amplitude_nonpref)
    Ext_input_stim1_pool2 = generate_Ext_input(T, N - N//2, dt, sigma, tau, amplitude_nonpref)
    Ext_input_stim2_pool2 = generate_Ext_input(T, N - N//2, dt, sigma, tau, amplitude_pref)
    Ext_input_stim1 = np.concatenate((Ext_input_stim1_pool1, Ext_input_stim1_pool2), axis=1)[None]
    Ext_input_stim2 = np.concatenate((Ext_input_stim2_pool1, Ext_input_stim2_pool2), axis=1)[None]
    Ext_input = np.concatenate((Ext_input_stim1, Ext_input_stim2), axis=0)
    return Ext_input

def Perturbation_input(Nstimuli, T, N, dt, trial_type, which_neurons, which_times, strength):
    Pert_input = np.zeros((Nstimuli, T, N))
    VN = np.zeros(N)
    VN[which_neurons] = strength
    t = dt * np.arange(T)
    VT = ( (t>which_times[0]) * (t<which_times[1]) ).astype(int)
    Pert_input[trial_type] = np.outer(VT,VN)
    return Pert_input

def sequence(t,N,width):
    T = len(t)
    Peak_times = np.linspace(0, t[-1], N)
    R = np.empty((T, N))
    for i in range(N):
        R[:, i] = np.exp(-(t - Peak_times[i]) ** 2 / (2 * width ** 2))
        R[:, i] = R[:, i] / max(R[:, i])
    return R

def two_sequences(t,N,width):
    T = len(t)
    Seq_pool1 = sequence(t, N//2, width)
    NoSeq_pool1 = np.zeros((T,N//2))
    Seq_pool2 = sequence(t, N-N//2, width)
    NoSeq_pool2 = np.zeros((T, N - N//2))
    activity1 = np.concatenate((Seq_pool1, NoSeq_pool2), axis=1)[None]
    activity2 = np.concatenate((NoSeq_pool1, Seq_pool2), axis=1)[None]
    return np.concatenate((activity1, activity2), axis=0)

def two_sequences_EI(t, N_E, N_I, width):
    T = len(t)
    Npool = N_E+N_I
    Seq_E = sequence(t, N_E, width)
    Seq_I = sequence(t, N_I, width)
    Seq = np.concatenate((Seq_E, Seq_I), axis=1)
    NoSeq = np.zeros((T, Npool))
    activity1 = np.concatenate((Seq, NoSeq), axis=1)[None]
    activity2 = np.concatenate((NoSeq, Seq), axis=1)[None]
    return np.concatenate((activity1, activity2), axis=0)

def get_xy_for_bandDiag_plot(J):
    N = J.shape[0]
    Diags = np.empty(2*N-1)
    for i,offset in enumerate(np.arange(-N+1, N)):
        Diags[i] = np.nanmean(np.diagonal(J, offset))
    x = np.arange(-N+1, N)[::-1]
    y = Diags
    return x,y

# def make_sequential_activity(t, N_sel, N_un, width, background_activity=0):
#     T = len(t)
#     Seq_sel = sequence(t, N_sel, width)
#     Seq_un = sequence(t, N_un, width)
#     NoSeq_sel = np.zeros((T,N_sel))
#     activity1 = np.concatenate((Seq_sel, NoSeq_sel, Seq_un), axis=1)[None]
#     activity2 = np.concatenate((NoSeq_sel, Seq_sel, Seq_un), axis=1)[None]
#     return np.concatenate((activity1, activity2), axis=0) + background_activity

def get_band_matrix(matrix, k, which):
    if which=='in':
        upper_triangular = np.triu(matrix, k=-k)
        lower_triangular = np.tril(matrix, k=k)
        band_matrix = upper_triangular * lower_triangular
    if which=='out':
        upper_triangular = np.triu(matrix, k=k)
        lower_triangular = np.tril(matrix, k=-k)
        band_matrix = upper_triangular + lower_triangular
    return band_matrix

def my_relu(x,T):
    return (x-T)*(x-T>0)
##

