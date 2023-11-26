import sys
sys.path.insert(0,'./modules')
from RecurrentNets import *
from decoding import *
from numpy import *
from dynamicstools import *
from plottingtools import *
from matplotlib.pyplot import *
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('ticks')
import rc_parameters
from numpy.linalg import multi_dot, inv, eigvals
from sklearn.metrics import roc_auc_score, auc
from scipy.stats import norm
from scipy import linalg
rcParams.update({"text.usetex": False, "axes.spines.top":False, "axes.spines.right":False})

## Set up dynamics model

S_EE = 0
S_EI = 2
S_IE = 2
S_II = 0.

D_EE = 0.
D_EI = -0.7
D_IE = 1
D_II =0

S = array([[S_EE, -S_EI],[S_IE, -S_II]])
D = array([[D_EE, -D_EI],[D_IE, -D_II]])
Win = 0.5 * (S + D)
Wout = 0.5 * (S - D)
J = block([[Win, Wout],[Wout, Win]])

DT = 20
T = 200 + DT
T1,T2 = 70,160
nsteps = 2000
t = linspace(0,T,nsteps)
dt = t[1] - t[0]
N = 4
baseline = 0
c1 = 1
c2 = 0
mu_ns = 0
input = outer(array([baseline,0,baseline,0]), ones(nsteps))
input = input + outer(array([c1,0,c2,0]), ones(nsteps)*(t>T1)*(t<T2))
pars = {'N': N, 't':t, 'R0':zeros(N), 'J':J, 'I':input, 'sigma':0., 'f': lambda x:x}
R = simulate_rate_model(pars)

##


# Fig. 4a: example_dynamics
fig,ax = my_figure(figsize=(1.8,1.5))
times = arange(int(DT/dt),len(t))

ax.plot(t[times]-DT, R[0,times], color='#4b669c', lw=1, label='E1') # E1
ax.plot(t[times]-DT,R[2,times], color='#4ea9ad', lw=1, label='E2') # E2

ax.set_xlim(0,T-DT)
ylim([-2,2.2])
y2 = ax.get_ylim()[1]
#ax.plot([T1-DT,T2-DT], [y2-0.05, y2-0.05], lw=2.5, color='#777a7a')

plot([0,T1-DT],[0.0,0.0], '--', lw=0.7, c='#4b669c')
plot([T1-DT,T1-DT],[0,1], '--', lw=0.7, c='#4b669c')
plot([T1-DT,T2-DT],[1,1], '--', lw=0.7, c='#4b669c')
plot([T2-DT,T2-DT],[1,0], '--', lw=0.7, c='#4b669c')
plot([T2-DT,T2],[0,0], '--', lw=0.7, c='#4b669c')

plot([0,T2],[0,0], '--', lw=0.7, c='#4ea9ad')

xlabel('Time (a.u.)')
#ylabel('Firing rate \nchange (a.u.)')
ylabel('Act. Change (Arb.)')
#legend(frameon=False)
tight_layout()

##

# Set up steady state model
S_EE = 0.
S_EI = 2
S_IE = 2
S_II = 0.

n = 200
D_EE = S_EE
D_EI = linspace(-2, 2, n)
D_IE = array([0.4,1,1.6])
D_II = 0.

amplification = empty((len(D_IE),len(D_EI)))
suppression = empty((len(D_IE),len(D_EI)))
distance = empty((len(D_IE),len(D_EI)))

for i_EI in range(len(D_EI)):
    for i_IE in range(len(D_IE)):

        if 1 + D_EI[i_EI] * D_IE[i_IE] - D_EE < 1e-10:
            amplification[i_IE, i_EI] = nan
            suppression[i_IE, i_EI]   = nan
            distance[i_IE, i_EI] = nan
        else:
            alpha = 1 - S_EE + S_EI * S_IE
            delta = 1 - D_EE + D_EI[i_EI] * D_IE[i_IE]

            amplification[i_IE, i_EI] = (1 / alpha + 1 / delta) / 2
            suppression[i_IE, i_EI] = (1 / alpha - 1 / delta) / 2
            distance[i_IE, i_EI] = 1 / delta

##

# Fig 4c: steady_state_E1_E2
fig = figure(figsize=(1.4,1.8)) #smalle (v4)
# Suppression
ax = fig.add_subplot(2,1,2)
format_axes(ax,'0')
col = ['#b3b3b3', '#6d757d', '#35424f']
labels = [r'$\Delta_{IE} = 0.6$', r'$\Delta_{IE} = 1$', r'$\Delta_{IE} = 1.4$']
for i in range(len(D_IE)):
    ax.plot(D_EI, suppression[i,:], lw=1., color=col[i], label=labels[i])
    ax.plot([-1.5,1.5],[1,1],'--', lw=0.7, color='grey')
    ax.set_ylim(-2,0)
    ax.set_yticks([0,-1,-2])
    ax.set_xlim(-1.5,1.5)
    ax.set_ylabel(r'E$_R$')
    ax.set_xlabel('I-to-E selectivity')
tight_layout()

# Amplification
ax = fig.add_subplot(2,1,1)
format_axes(ax,'0')
# col = ['#8aaced', '#4775cc', '#073894']
# labels = [r'$\Delta_{IE} = 0.6$', r'$\Delta_{IE} = 1$', r'$\Delta_{IE} = 1.4$']
labels = [r'E-to-I sel. = 0.6', r'E-to-I sel. = 1', r'E-to-I sel. = 1.4']
for i in range(len(D_IE)):
    ax.plot(D_EI, amplification[i,:], lw=1., color=col[i], label=labels[i])
    ax.plot([-1.5,1.5],[1,1],'--', lw=0.7, color='grey')
    ax.set_ylim(0,2)
    ax.set_xlim(-1.5,1.5)
    ax.set_xticks([])
    ax.set_yticks([0,1,2])
    ax.set_ylabel(r'E$_L$')
# legend(frameon=0, fontsize=9)
tight_layout()


## Fig 4d: distance of mean activities
# fig,ax=my_figure(figsize=(1.9,1.6))
fig,ax=my_figure(figsize=(1.4,1.2))

for i in range(len(D_IE)):
    ax.plot(D_EI, distance[i,:], lw=1., color=col[i])
ax.set_ylim(0,2)
ax.set_xlim(-1.5,1.5)

ax.plot([0,0],[0,2],'--', lw=0.7, color='grey')
ax.plot([-1.5,1.5],[1,1],'--', lw=0.7, color='grey')

ax.set_xlabel(r'I-to-E selectivity')
ax.set_ylabel('Dist. (norm.)')
tight_layout()

##

# Set up decoding accuracy model
n = 120 # fine grain ness of model (was 120)
S_EE = 0
S_EI = 2
S_IE = 2
S_II = 0.

D_EE = S_EE
D_EI = linspace(-S_EI, S_EI, n)
D_IE = linspace(0, S_IE, n)
D_II = 0.

decodacc_E     = empty((len(D_IE),len(D_EI)))
decodacc_EI    = empty((len(D_IE),len(D_EI)))
decodacc_E_sh  = empty((len(D_IE),len(D_EI)))
decodacc_EI_sh = empty((len(D_IE),len(D_EI)))

decodacc_E_0     = empty((len(D_IE),len(D_EI)))
decodacc_EI_0    = empty((len(D_IE),len(D_EI)))
decodacc_E_sh_0  = empty((len(D_IE),len(D_EI)))
decodacc_EI_sh_0 = empty((len(D_IE),len(D_EI)))

amplification = empty((len(D_IE),len(D_EI)))
suppression = empty((len(D_IE),len(D_EI)))

N = 4
sigma_input = 0
sigma_obs = 1
R0 = zeros(N)
f = lambda x:x

for i_EI in range(len(D_EI)):
    for i_IE in range(len(D_IE)):

        if 1 + D_EI[i_EI] * D_IE[i_IE] - D_EE < 1e-10:
            decodacc_EI[i_IE, i_EI]     = nan
            decodacc_E[i_IE, i_EI]      = nan
            decodacc_EI_sh[i_IE, i_EI]  = nan
            decodacc_E_sh[i_IE, i_EI]   = nan

        else:
            # set connectivity matrix
            S    = array([[S_EE, -S_EI], [S_IE, -S_II]])
            D    = array([[D_EE, -D_EI[i_EI]], [D_IE[i_IE], -D_II]])
            Win  = 0.5 * (S + D)
            Wout = 0.5 * (S - D)

            alpha = 1-S_EE+S_EI*S_IE
            delta = 1-D_EE+D_EI[i_EI]*D_IE[i_IE]

            amplification[i_IE, i_EI] = (1/alpha+1/delta)/2
            suppression[i_IE, i_EI] = (1/alpha-1/delta)/2

            J    = block([[Win, Wout], [Wout, Win]])
            Net = RecurrentNet(J, sigma_input)
            _ = Net.covariance()
            dh1 = array([1,0,0,0])
            dh2 = array([0,0,1,0])
            dh = dh1 - dh2

            cov = sigma_obs * eye(N)
            decodacc_EI[i_IE, i_EI], decodacc_EI_sh[i_IE, i_EI] = Net.compute_acc_analytical_obsnoise(dh, cov)
            decodacc_E[i_IE, i_EI], decodacc_E_sh[i_IE, i_EI] = Net.compute_acc_analytical_obsnoise(dh, cov, idx = [0, 2])

            J = zeros((N,N))
            Net = RecurrentNet(J, sigma_input)
            _ = Net.covariance()
            decodacc_EI_0[i_IE, i_EI], decodacc_EI_sh_0[i_IE, i_EI] = Net.compute_acc_analytical_obsnoise(dh, cov)
            decodacc_E_0[i_IE, i_EI], decodacc_E_sh_0[i_IE, i_EI] = Net.compute_acc_analytical_obsnoise(dh, cov, idx=[0, 2])


## Fig. 4e: decoding accuracy phase plot
DD_EI, DD_IE = meshgrid(D_EI, D_IE)
delta = 1 - D_EE + DD_EI*DD_IE

#fig,ax = my_figure(figsize=(2.4,1.9))
fig,ax = my_figure(figsize=(2.8,2))
colors = ['w','#750b40']
cm = define_colormap(colors,100)
Z = decodacc_E

Z[delta<0] = nan
cm.set_bad(color='#d4d4d4')
clim = (0.5,1)
c = ax.imshow(Z,cmap=cm, clim=clim, extent = [min(D_EI),max(D_EI),min(D_IE),max(D_IE)], origin = 'lower')


x1 = linspace(-2,(D_EE-1)/2,1000)
x2 = linspace(-(D_EE-1)/2,2,1000)
ax.plot(x1, (D_EE-1)/x1, lw=1, color='#48494a')

xlim = (-2,2)
ylim = (0,2)

xlabel('I-to-E selectivity')
ylabel('E-to-I \n selectivity')
plot([0,0],[0,S_IE],'-',color='k', lw=.7)
plot([-2,2],[1,1], lw=0.6, ls='--', color='k')
plot([-.7,-.7],[0,2], lw=0.6, ls='--', color='k')
plot([-0.7],[1],'s', markersize=4, markeredgecolor='w', markeredgewidth=0.5, color='k')

#cbar = fig.colorbar(c, ticks=(0.5,1), fraction=0.025).set_label('Decod. \n accuracy')
# Put colorbar on right
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
ax_divider = make_axes_locatable(ax)
# add an axes above the main axes.
cax = ax_divider.append_axes("right", size="7%", pad="5%")
cb = fig.colorbar(c, cax=cax, orientation="vertical",ticks=(0.5,1)).set_label('Decod. \n accuracy')


tight_layout()


## Fig. 4e: rel decoding accuracy phase plot

DD_EI, DD_IE = meshgrid(D_EI, D_IE)
delta = 1 - D_EE + DD_EI*DD_IE

fig,ax = my_figure(figsize=(2.4,1.9))
colors = ['#086096','w','#c93a60','#851835']
cm = define_colormap(colors,100)

Z = abs((decodacc_E-0.5)/(decodacc_E_0-0.5))

Z[delta<0] = nan
cm.set_bad(color='#d4d4d4')

clim = (0,3)
c = ax.imshow(Z,cmap=cm, clim=clim, extent = [min(D_EI),max(D_EI),min(D_IE),max(D_IE)], origin = 'lower')

x1 = linspace(-2,(D_EE-1)/2,1000)
x2 = linspace(-(D_EE-1)/2,2,1000)
ax.plot(x1, (D_EE-1)/x1, lw=1, color='#48494a')

xlim = (-2,2)
ylim = (0,2)

xlabel('I-to-E selectivity')
ylabel('E-to-I \n selectivity')
plot([0,0],[0,S_IE],'-',color='k', lw=.7)
plot([-2,2],[1,1], lw=0.6, ls='--', color='k')
plot([-1,-1],[0,2], lw=0.6, ls='--', color='k')
plot([-0.7],[1],'s', markersize=4, markeredgecolor='w', markeredgewidth=0.5, color='k')

# Put colorbar on right
# cbar = fig.colorbar(c, ticks=(0,1,2,3), fraction=0.025).set_label('Rel. decod. \n accuracy')

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
ax_divider = make_axes_locatable(ax)
# add an axes above the main axes.
cax = ax_divider.append_axes("right", size="7%", pad="5%")
cb = fig.colorbar(c, cax=cax, orientation="vertical",ticks=(0,1,2,3)).set_label('Rel. Decod. \n accuracy')
tight_layout()

##


# Set up I-to-E cuts
S_EE = 0.
S_EI = 2
S_IE = 2
S_II = 0.

n = 200
D_EE = S_EE
# D_EI = linspace(-S_EI, S_EI, n)
D_EI = linspace(-2, 2, n)
D_IE = array([1])
D_II = 0.

decodacc_E       = empty((len(D_IE),len(D_EI)))
decodacc_EI      = empty((len(D_IE),len(D_EI)))
decodacc_E1      = empty((len(D_IE),len(D_EI)))
decodacc_E2      = empty((len(D_IE),len(D_EI)))
decodacc_I1      = empty((len(D_IE),len(D_EI)))
decodacc_I2      = empty((len(D_IE),len(D_EI)))

decodacc_E_0     = empty((len(D_IE),len(D_EI)))
decodacc_EI_0    = empty((len(D_IE),len(D_EI)))
decodacc_E1_0    = empty((len(D_IE),len(D_EI)))
decodacc_E2_0    = empty((len(D_IE),len(D_EI)))
decodacc_I1_0    = empty((len(D_IE),len(D_EI)))
decodacc_I2_0    = empty((len(D_IE),len(D_EI)))


N = 4
sigma_input = 0
sigma_obs = 2
R0 = zeros(N)
f = lambda x:x

for i_EI in range(len(D_EI)):
    for i_IE in range(len(D_IE)):

        if 1 + D_EI[i_EI] * D_IE[i_IE] - D_EE < 1e-10:
            decodacc_EI[i_IE, i_EI]     = nan
            decodacc_E[i_IE, i_EI]      = nan
            decodacc_EI_0[i_IE, i_EI]   = nan
            decodacc_E_0[i_IE, i_EI]    = nan

            decodacc_E1[i_IE, i_EI]   = nan
            decodacc_E2[i_IE, i_EI]   = nan
            decodacc_E1_0[i_IE, i_EI] = nan
            decodacc_E2_0[i_IE, i_EI] = nan

            decodacc_I1[i_IE, i_EI]   = nan
            decodacc_I2[i_IE, i_EI]   = nan
            decodacc_I1_0[i_IE, i_EI] = nan
            decodacc_I2_0[i_IE, i_EI] = nan

        else:
            S    = array([[S_EE, -S_EI], [S_IE, -S_II]])
            D    = array([[D_EE, -D_EI[i_EI]], [D_IE[i_IE], -D_II]])
            Win  = 0.5 * (S + D)
            Wout = 0.5 * (S - D)

            J    = block([[Win, Wout], [Wout, Win]])
            if max(eigvals(J).real)>1:
                print('Unstable!')

            Net = RecurrentNet(J, sigma_input)
            _ = Net.covariance()
            dh1 = array([1,0,0,0])
            dh2 = array([0,0,1,0])
            dh = dh1 - dh2

            cov = sigma_obs**2 * eye(N)

            decodacc_EI[i_IE, i_EI], _ = Net.compute_acc_analytical_obsnoise(dh, cov)
            decodacc_E[i_IE, i_EI],  _ = Net.compute_acc_analytical_obsnoise(dh, cov, idx = [0,2])

            J = zeros((N,N))
            Net = RecurrentNet(J, sigma_input)
            _ = Net.covariance()
            decodacc_EI_0[i_IE, i_EI], _ = Net.compute_acc_analytical_obsnoise(dh, cov)
            decodacc_E_0[i_IE, i_EI],  _ = Net.compute_acc_analytical_obsnoise(dh, cov, idx=[0, 2])

##
# set parameters
T = 200
nsteps = 2000
t = linspace(0, T, nsteps)
N = 4
ntrials = 5000
R0 = zeros(N)
f = lambda x:x


nsim = 15
D_EI = linspace(D_EI[0], D_EI[-1], nsim)

decodacc_E_sim       = empty((len(D_IE),len(D_EI)))
decodacc_EI_sim      = empty((len(D_IE),len(D_EI)))
decodacc_E1_sim      = empty((len(D_IE),len(D_EI)))
decodacc_E2_sim      = empty((len(D_IE),len(D_EI)))
decodacc_I1_sim      = empty((len(D_IE),len(D_EI)))
decodacc_I2_sim      = empty((len(D_IE),len(D_EI)))

h1 = array([1,0,0,0])
h2 = array([0,0,1,0])
h1 = h1 / linalg.norm(h1-h2)
h2 = h2 / linalg.norm(h1-h2)

# compute decoding accuracy WITHOUT recurrent connections using single-trial simulations
input = outer(h1, ones(nsteps))
pars = {'N': N, 't':t, 'R0':zeros(N), 'J':zeros((N,N)), 'I':input, 'sigma':sigma_input, 'f': lambda x:x}
R = simulate_rate_model(pars)
fp_1 = R[:,-1]
R_singletrials_1 = outer(fp_1, ones(ntrials)) + random.normal(0, sigma_obs, (N, ntrials))

# set second input and compute fixed point
input = outer(h2, ones(nsteps))
pars = {'N': N, 't': t, 'R0': zeros(N), 'J': zeros((N,N)), 'I': input, 'sigma': sigma_input, 'f': lambda x:x}
R = simulate_rate_model(pars)
fp_2 = R[:, -1]
R_singletrials_2 = outer(fp_2, ones(ntrials)) + random.normal(0, sigma_obs, (N, ntrials))

S = hstack((ones(ntrials), -ones(ntrials)))
R = hstack((R_singletrials_1, R_singletrials_2))
clf = LinearSVM(R,S)
clf_E = LinearSVM(R[[0,2],:], S)
clf.set_K(5)
clf_E.set_K(5)
decodacc_EI_0_sim  = mean(clf.get_accuracy())
decodacc_E_0_sim   = mean(clf_E.get_accuracy())


# compute decoding accuracy WITH recurrent connections using single-trial simulations
for i_IE in range(len(D_IE)):
    for i_EI in range(len(D_EI)):
        print(i_EI)

        # set connectivity matrix
        S = array([[S_EE, -S_EI], [S_IE, -S_II]])
        D = array([[D_EE, -D_EI[i_EI]], [D_IE[i_IE], -D_II]])
        Win = 0.5 * (S + D)
        Wout = 0.5 * (S - D)
        J = block([[Win, Wout], [Wout, Win]])
        d = max(eigvals(J).real)

        if 1+D_EI[i_EI]*D_IE[i_IE]-D_EE < 1e-10:
            decodacc_EI_sim[i_IE, i_EI] = nan
            decodacc_E_sim[i_IE, i_EI]  = nan
            decodacc_E1_sim[i_IE, i_EI] = nan
            decodacc_E2_sim[i_IE, i_EI] = nan
            decodacc_I1_sim[i_IE, i_EI] = nan
            decodacc_I2_sim[i_IE, i_EI] = nan

        else:
            print(d)

            # set first input and compute fixed point
            input = outer(h1, ones(nsteps))
            pars = {'N': N, 't':t, 'R0':zeros(N), 'J':J, 'I':input, 'sigma':sigma_input, 'f': lambda x:x}
            R = simulate_rate_model(pars)
            fp_1 = R[:,-1]
            R_singletrials_1 = outer(fp_1, ones(ntrials)) + random.normal(0, sigma_obs, (N, ntrials))

            # set second input and compute fixed point
            input = outer(h2, ones(nsteps))
            pars = {'N': N, 't': t, 'R0': zeros(N), 'J': J, 'I': input, 'sigma': sigma_input, 'f': lambda x:x}
            R = simulate_rate_model(pars)
            fp_2 = R[:, -1]
            R_singletrials_2 = outer(fp_2, ones(ntrials)) + random.normal(0, sigma_obs, (N, ntrials))


            S = hstack((ones(ntrials), -ones(ntrials)))
            R = hstack((R_singletrials_1, R_singletrials_2))
            clf = LinearSVM(R,S)
            clf_E = LinearSVM(R[[0,2],:], S)
            clf_E1 = LinearSVM(R[[0], :], S)
            clf_E2 = LinearSVM(R[[2],:], S)
            clf_I1 = LinearSVM(R[[1], :], S)
            clf_I2 = LinearSVM(R[[3], :], S)
            clf.set_K(10)
            clf_E.set_K(10)
            clf_E1.set_K(10)
            clf_E2.set_K(10)
            clf_I1.set_K(10)
            clf_I2.set_K(10)
            decodacc_EI_sim[i_IE, i_EI]  = mean(clf.get_accuracy())
            decodacc_E_sim[i_IE, i_EI]   = mean(clf_E.get_accuracy())

##

# I-E cut decoding accuracy
figsize=(1.6,1.1)

fig = figure(figsize=figsize)
ax = fig.add_subplot(1,1,1)
format_axes(ax, '0')

black = '#313333'

#limx = [-2,2]
limx = [-1.2,1.2]
limy1,limy2 = 0.5,1.025

n = 200
D_EI_1 = linspace(D_EI[0], D_EI[-1], n)
Lim = (D_EE-1)/D_IE
D_EI_2 = linspace(D_EI[0], D_EI[-1], nsim)

colors = ['w','#750b40']
cm = define_colormap(colors,100)
for i in range(len(D_IE)):
    plot(D_EI_1, decodacc_E[i,:], lw=0.7, color=black)
    #plot(D_EI_2, decodacc_E_sim[i,:], '.', \
    #     markeredgewidth=0.2, markeredgecolor='k', markersize=4,               color='#e85177')
    ysim = decodacc_E_sim[i,:]
    plt.scatter(D_EI_2, ysim, c=ysim, s=10, linewidths = 0.5, edgecolors='k',cmap=cm)

plot([0,0],[0,3],'--',lw=0.6, color='grey')
a = abs(D_EI[0])
fill_between([-2,-1], [limy1, limy1], [limy2,limy2], color='#d4d4d4')

xlabel('I-to-E selectivity')
ylabel('Decod. \naccuracy')
plt.xlim(limx)
yticks([0.5,1])
ax.set_ylim(limy1,limy2)
#xticks([-2,-1,0,1,2])

tight_layout()

##


# I-E cut rel decoding accuracy
figsize=(1.6,1.1)
fig = figure(figsize=figsize)
ax = fig.add_subplot(1,1,1)
format_axes(ax, '0')
black = '#313333'
colors = ['#086096','w','#c93a60','#851835']
cm = define_colormap(colors,100)

#limx = [0,2]
limx = [-1.2, 1.2]
#limy1,limy2 = 0.,3.
limy1,limy2 = 0.,5.2

n = 200
D_EI_1 = linspace(D_EI[0], D_EI[-1], n)
Lim = (D_EE-1)/D_IE
D_EI_2 = linspace(D_EI[0], D_EI[-1], nsim)

plot([0,0],[0,limy2],'--',lw=0.6, color='grey')
plot([-2,2],[1,1],'--',lw=0.6, color='grey')

for i in range(len(D_IE)):
    plot(D_EI_1, ((decodacc_E - 0.5) / (decodacc_E_0 - 0.5))[i,:], lw=0.5, color=black)
    #plot(D_EI_2, ((decodacc_E_sim - 0.5) / (decodacc_E_0_sim - 0.5))[i,:], '.', \
         #markeredgewidth=0.2, markeredgecolor='k', markersize=4, color='#e85177')
    ysim = ((decodacc_E_sim - 0.5) / (decodacc_E_0_sim - 0.5))[i,:]
    plt.scatter(D_EI_2, ysim, c=ysim, s=10, linewidths = 0.5, edgecolors='k',cmap=cm)

a = abs(D_EI[0])
fill_between([-2,-1], [limy1, limy1], [limy2,limy2], color='#d4d4d4')

xlabel('I-to-E selectivity')
ylabel('Rel. decod. \naccuracy')
ax.set_xlim(limx)
#yticks([0,1,2,3])
yticks([0,limy2])
yticks([0,5])
ax.set_ylim([limy1,limy2])
#xticks([-2,-1,0,1,2])
xticks([-1,0,1])
tight_layout()


##

# setup E-to-I cuts
S_EE = 0.
S_EI = 2
S_IE = 2
S_II = 0.

n = 200
D_EE = S_EE
# D_EI = linspace(-S_EI, S_EI, n)
D_EI = array([-1])
D_IE = linspace(0, 2, n)
D_II = 0.

decodacc_E       = empty((len(D_IE),len(D_EI)))
decodacc_EI      = empty((len(D_IE),len(D_EI)))
decodacc_E1      = empty((len(D_IE),len(D_EI)))
decodacc_E2      = empty((len(D_IE),len(D_EI)))
decodacc_I1      = empty((len(D_IE),len(D_EI)))
decodacc_I2      = empty((len(D_IE),len(D_EI)))

decodacc_E_0     = empty((len(D_IE),len(D_EI)))
decodacc_EI_0    = empty((len(D_IE),len(D_EI)))
decodacc_E1_0    = empty((len(D_IE),len(D_EI)))
decodacc_E2_0    = empty((len(D_IE),len(D_EI)))
decodacc_I1_0    = empty((len(D_IE),len(D_EI)))
decodacc_I2_0    = empty((len(D_IE),len(D_EI)))


N = 4
sigma_input = 0
sigma_obs = 2
R0 = zeros(N)
f = lambda x:x

for i_EI in range(len(D_EI)):
    for i_IE in range(len(D_IE)):

        if 1 + D_EI[i_EI] * D_IE[i_IE] - D_EE < 1e-10:
            decodacc_EI[i_IE, i_EI]     = nan
            decodacc_E[i_IE, i_EI]      = nan
            decodacc_EI_0[i_IE, i_EI]   = nan
            decodacc_E_0[i_IE, i_EI]    = nan

            decodacc_E1[i_IE, i_EI]   = nan
            decodacc_E2[i_IE, i_EI]   = nan
            decodacc_E1_0[i_IE, i_EI] = nan
            decodacc_E2_0[i_IE, i_EI] = nan

            decodacc_I1[i_IE, i_EI]   = nan
            decodacc_I2[i_IE, i_EI]   = nan
            decodacc_I1_0[i_IE, i_EI] = nan
            decodacc_I2_0[i_IE, i_EI] = nan

        else:
            S    = array([[S_EE, -S_EI], [S_IE, -S_II]])
            D    = array([[D_EE, -D_EI[i_EI]], [D_IE[i_IE], -D_II]])
            Win  = 0.5 * (S + D)
            Wout = 0.5 * (S - D)

            J    = block([[Win, Wout], [Wout, Win]])
            if max(eigvals(J).real)>1:
                print('Unstable!')

            Net = RecurrentNet(J, sigma_input)
            _ = Net.covariance()
            dh1 = array([1,0,0,0])
            dh2 = array([0,0,1,0])
            dh = dh1 - dh2

            cov = sigma_obs**2 * eye(N)

            decodacc_EI[i_IE, i_EI], _ = Net.compute_acc_analytical_obsnoise(dh, cov)
            decodacc_E[i_IE, i_EI],  _ = Net.compute_acc_analytical_obsnoise(dh, cov, idx = [0,2])

            J = zeros((N,N))
            Net = RecurrentNet(J, sigma_input)
            _ = Net.covariance()
            decodacc_EI_0[i_IE, i_EI], _ = Net.compute_acc_analytical_obsnoise(dh, cov)
            decodacc_E_0[i_IE, i_EI],  _ = Net.compute_acc_analytical_obsnoise(dh, cov, idx=[0, 2])

##
# set parameters
T = 200
nsteps = 2000
t = linspace(0, T, nsteps)
N = 4
ntrials = 5000
R0 = zeros(N)
f = lambda x:x


nsim = 15
D_IE = linspace(D_IE[0], D_IE[-1], nsim)

decodacc_E_sim       = empty((len(D_IE),len(D_EI)))
decodacc_EI_sim      = empty((len(D_IE),len(D_EI)))

h1 = array([1,0,0,0])
h2 = array([0,0,1,0])
h1 = h1 / linalg.norm(h1-h2)
h2 = h2 / linalg.norm(h1-h2)

# compute decoding accuracy WITHOUT recurrent connections using single-trial simulations
input = outer(h1, ones(nsteps))
pars = {'N': N, 't':t, 'R0':zeros(N), 'J':zeros((N,N)), 'I':input, 'sigma':sigma_input, 'f': lambda x:x}
R = simulate_rate_model(pars)
fp_1 = R[:,-1]
R_singletrials_1 = outer(fp_1, ones(ntrials)) + random.normal(0, sigma_obs, (N, ntrials))

# set second input and compute fixed point
input = outer(h2, ones(nsteps))
pars = {'N': N, 't': t, 'R0': zeros(N), 'J': zeros((N,N)), 'I': input, 'sigma': sigma_input, 'f': lambda x:x}
R = simulate_rate_model(pars)
fp_2 = R[:, -1]
R_singletrials_2 = outer(fp_2, ones(ntrials)) + random.normal(0, sigma_obs, (N, ntrials))

S = hstack((ones(ntrials), -ones(ntrials)))
R = hstack((R_singletrials_1, R_singletrials_2))
clf = LinearSVM(R,S)
clf_E = LinearSVM(R[[0,2],:], S)
clf.set_K(5)
clf_E.set_K(5)
decodacc_EI_0_sim  = mean(clf.get_accuracy())
decodacc_E_0_sim   = mean(clf_E.get_accuracy())


# compute decoding accuracy WITH recurrent connections using single-trial simulations
for i_IE in range(len(D_IE)):
    for i_EI in range(len(D_EI)):
        print(i_EI)

        # set connectivity matrix
        S = array([[S_EE, -S_EI], [S_IE, -S_II]])
        D = array([[D_EE, -D_EI[i_EI]], [D_IE[i_IE], -D_II]])
        Win = 0.5 * (S + D)
        Wout = 0.5 * (S - D)
        J = block([[Win, Wout], [Wout, Win]])
        d = max(eigvals(J).real)

        if 1+D_EI[i_EI]*D_IE[i_IE]-D_EE < 1e-10:
            decodacc_EI_sim[i_IE, i_EI] = nan
            decodacc_E_sim[i_IE, i_EI]  = nan

        else:
            print(d)

            # set first input and compute fixed point
            input = outer(h1, ones(nsteps))
            pars = {'N': N, 't':t, 'R0':zeros(N), 'J':J, 'I':input, 'sigma':sigma_input, 'f': lambda x:x}
            R = simulate_rate_model(pars)
            fp_1 = R[:,-1]
            R_singletrials_1 = outer(fp_1, ones(ntrials)) + random.normal(0, sigma_obs, (N, ntrials))

            # set second input and compute fixed point
            input = outer(h2, ones(nsteps))
            pars = {'N': N, 't': t, 'R0': zeros(N), 'J': J, 'I': input, 'sigma': sigma_input, 'f': lambda x:x}
            R = simulate_rate_model(pars)
            fp_2 = R[:, -1]
            R_singletrials_2 = outer(fp_2, ones(ntrials)) + random.normal(0, sigma_obs, (N, ntrials))


            S = hstack((ones(ntrials), -ones(ntrials)))
            R = hstack((R_singletrials_1, R_singletrials_2))
            clf = LinearSVM(R,S)
            clf_E = LinearSVM(R[[0,2],:], S)
            clf_E1 = LinearSVM(R[[0], :], S)
            clf_E2 = LinearSVM(R[[2],:], S)
            clf_I1 = LinearSVM(R[[1], :], S)
            clf_I2 = LinearSVM(R[[3], :], S)
            clf.set_K(10)
            clf_E.set_K(10)
            clf_E1.set_K(10)
            clf_E2.set_K(10)
            clf_I1.set_K(10)
            clf_I2.set_K(10)
            decodacc_EI_sim[i_IE, i_EI]  = mean(clf.get_accuracy())
            decodacc_E_sim[i_IE, i_EI]   = mean(clf_E.get_accuracy())

##


# I-E cut decoding accuracy
figsize=(1.7,1.1)
fig = figure(figsize=figsize)
ax = fig.add_subplot(1,1,1)
format_axes(ax, '0')
black = '#313333'

limx = [-0.05,1.2]
limy1,limy2 = 0.5,1.

n = 200
D_IE_1 = linspace(D_IE[0], D_IE[-1], n)
Lim = (D_EE-1)/D_EI
D_IE_2 = linspace(D_IE[0], D_IE[-1], nsim)

colors = ['w','#750b40']
cm = define_colormap(colors,100)
for i in range(len(D_EI)):
    plot(D_IE_1, decodacc_E[:, i], lw=0.7, color=black)
    #plot(D_IE_2, decodacc_E_sim[:, i], '.', \
    #     markeredgewidth=0.2, markeredgecolor='k', markersize=4, color='#0db0d1')
    ysim = decodacc_E_sim[:, i]
    plt.scatter(D_IE_2, ysim, c=ysim, s=10, linewidths = 0.5, edgecolors='k',cmap=cm)

a = abs(D_EI[0])
fill_between([Lim[0], 2], [limy1, limy1], [limy2,limy2], color='#d4d4d4')

xlabel('E-to-I selectivity')
ylabel('Decod. \n accuracy')
plt.xlim(limx)
yticks([0.5,1])
plt.ylim([limy1,limy2])
#xticks([0,1,2])
tight_layout()

##


# I-E cut relative decoding accuracy
fig = figure(figsize=figsize)
#ax = fig.add_subplot(1,1,1)
ax = fig.add_subplot(1,1,1)
format_axes(ax, '0')

black = '#313333'
colors = ['#086096','w','#c93a60','#851835']
cm = define_colormap(colors,100)

#limx = [0,2]
limx = [-0.05,1.2]
limy1,limy2 = 0.,5.

n = 200
D_IE_1 = linspace(D_IE[0], D_IE[-1], n)
Lim = (D_EE-1)/D_EI
D_IE_2 = linspace(D_IE[0], D_IE[-1], nsim)


for i in range(len(D_EI)):
    plot(D_IE_1, ((decodacc_E - 0.5) / (decodacc_E_0 - 0.5))[:, i], lw=.7, color=black)
    #plot(D_IE_2, ((decodacc_E_sim - 0.5) / (decodacc_E_0_sim - 0.5))[:, i], '.', \
    #     markeredgewidth=0.2, markeredgecolor='k', markersize=4, color='#0db0d1')
    ysim = ((decodacc_E_sim - 0.5) / (decodacc_E_0_sim - 0.5))[:, i]
    plt.scatter(D_IE_2, ysim, c=ysim, s=10, linewidths = 0.5, edgecolors='k',cmap=cm)

plot([0,2],[1,1],'--',lw=0.6, color='grey')
a = abs(D_EI[0])
fill_between([Lim[0], 2], [limy1, limy1], [limy2,limy2], color='#d4d4d4')

xlabel(r'E-to-I selectivity')
ylabel('Rel. decod. \naccuracy')
plt.xlim(limx)
yticks([0,5])
plt.ylim([limy1,limy2])
#xticks([0,1,2])
tight_layout()



##


##


##

