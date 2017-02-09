"""

    Script for fitting the radial distribution function (RDF) obtained experimentally from cryo-Electron
    Tomography of Rubisco proteins in pyrenoids.

    Input:  - .pkl file with dependant and independents experimental function values
            - Function model
            - Fitting method

    Output: - Graphs with the fitting and the error

"""

__author__ = 'Antonio Martinez-Sanchez'

###### Global variables

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = './'

###### Input of the experimental estimations

in_pkl_x = ROOT_PATH + '/data/mean_RLD_x.pkl'
in_pkl_y = ROOT_PATH + '/data/mean_RLD_y.pkl'
in_pkl_cil = ROOT_PATH + '/data/ci_99_low.pkl'
in_pkl_cih = ROOT_PATH + '/data/ci_99_high.pkl'

####### Lennar Jonnes parameters

sigma = 13.9 # Length parameter of the LJ potential function

####### Fitting parameters

init_params = [1., 15] # Initial seed for rho* and T* LJ parameres
resamp_f = 1. # Re-sampling factor
# These bounds are only applied to estimate que error
min_r = 10.5 # nm
max_r = 60

####### Showing results

save = True # If None the graphs are plotted in block mode
# Y-axis limits
y_l, y_h = -.3, .8

####### Argon model parameters

rho_arg = 1.1
T_arg = 5.1

########################################################################################
# MAIN ROUTINE
########################################################################################

################# Package import

import time
import pickle
import warnings
import numpy as np
from scipy import signal
from scipy import optimize as opt
from models import LJ_Morsali, sqe_mean, sqe_max, sqe_std, ynorm
import matplotlib.pyplot as plt

########## Print initial message

print 'Lennard-Jones pure fluids RDF global fitting.'
print '\tAuthor: ' + __author__
print '\tDate: ' + time.strftime("%c") + '\n'
print 'Options:'
print '\tExperimental data:'
print '\t\tX-axis array pickle (radius in nm): ' + str(in_pkl_x)
print '\t\tY-axis array pickle (RLD): ' + str(in_pkl_y)
print '\tLJ parameters:'
print '\t\tSigma: ' + str(sigma) + ' nm'
print '\tArgon model: '
print '\t\t-Rho star: ' + str(rho_arg)
print '\t\t-Rho star: ' + str(T_arg)
print '\tFitting parameters: '
print '\t\t-Initial seed [rho*, T*]: ' + str(init_params)
print '\t\t-Resampling factor: ' + str(resamp_f)
print '\t\t-Fitting range: [' + str(min_r) + ', ' + str(max_r) + '] nm'
print '\tPlotting setting: '
if save:
    print '\t\t-Storing graphs directory: ' + ROOT_PATH + str('figs')
else:
    print '\t\t-Plotting graphs in block mode.'
print '\t\t-NRLD Y-axis limits: ' + str((y_l, y_h))
print ''

######### Process

print 'Main Routine: '

print '\tLoading pickles...'
exp_x = pickle.load(open(in_pkl_x, 'rb'))
exp_y = pickle.load(open(in_pkl_y, 'rb'))
exp_yl = pickle.load(open(in_pkl_cil, 'rb'))
exp_yh = pickle.load(open(in_pkl_cih, 'rb'))

print '\tResampling data: '
if resamp_f > 1:
    exp_y, exp_x = signal.resample(exp_y, int(resamp_f)*exp_x.shape[0], t=exp_x, window=('kaiser', 5.0))

print '\tDeleting samples lower than ' + str((min_r, max_r)) + ' nm...'
exp_x = exp_x[0:]
exp_y = exp_y[0:]
exp_yl = exp_yl[0:]
exp_yh = exp_yh[0:]
ids = np.where((exp_x >= min_r) & (exp_x <= max_r))[0]
exp_xc = exp_x[ids[0]:ids[-1]]
exp_yc = exp_y[ids[0]:ids[-1]]

print '\tInitializing the models...'
models = LJ_Morsali(sigma, exp_xc, exp_yc)

print '\tOptimization for fitting...'
warnings.filterwarnings('ignore')
obj_mat = opt.basinhopping(models.sqe_rdf, init_params, niter=10000, T=1.0, stepsize=0.01,
                           minimizer_kwargs=None, take_step=None, accept_test=None, callback=None,
                           interval=50, disp=False, niter_success=None)
found_params = obj_mat.x
print '\t\t-Parametrs found [rho*, T*]: ' + str(found_params)

print '\tFitting quality metrics: '
mat_rmse = np.sqrt(sqe_mean(models.rdf(exp_xc, found_params[0], found_params[1]), exp_yc))
mat_mse = sqe_mean(models.rdf(exp_xc, found_params[0], found_params[1]), exp_yc)
mat_max = sqe_max(models.rdf(exp_xc, found_params[0], found_params[1]), exp_yc)
mat_std = sqe_std(models.rdf(exp_xc, found_params[0], found_params[1]), exp_yc)
print '\t\t-Root squared mean error: ' + str(mat_rmse)
print '\t\t-Mean squared error: ' + str(mat_mse)
print '\t\t-Maximum squared error: ' + str(mat_max)
print '\t\t-Standard deviation of squared error: ' + str(mat_std)

print '\tPlotting results...'
plt.figure()
plt.title('Morsali model - Global fitting')
plt.xlabel('Radius (nm)')
plt.ylabel('RLD')
plt.plot(exp_x, exp_y, 'b')
rdl = models.rdf(exp_x, found_params[0], found_params[1])
plt.plot(exp_x, rdl, 'k')
plt.plot(exp_x, np.zeros(shape=len(exp_x)), 'k--')
if save:
    out_name = ROOT_PATH + '/figs/rld_s' + str(sigma) + '_r' + str(found_params[0]) + '_T' + \
               str(found_params[1]) + '.svg'
    plt.savefig(out_name, format='svg', dpi=1200)
else:
    plt.show(block=True)
plt.close()
plt.figure()
plt.title('Morsali model - Global fitting')
plt.xlabel('Radius (nm)')
plt.ylabel('NRLD')
plt.plot(exp_x, ynorm(exp_y, y_l, y_h), 'b')
plt.fill_between(exp_x, ynorm(exp_yl, y_l, y_h), ynorm(exp_yh, y_l, y_h), alpha=0.5, color='b',
                 edgecolor='w')
plt.plot(exp_x, ynorm(rdl, y_l, y_h), 'k')
plt.ylim((y_l, y_h))
plt.plot(exp_x, np.zeros(shape=len(exp_x)), 'k--')
if save:
    out_name = ROOT_PATH + '/figs/nrld_s' + str(sigma) + '_r' + str(found_params[0]) + '_T' + \
               str(found_params[1]) + '.svg'
    plt.savefig(out_name, format='svg', dpi=1200)
else:
    plt.show(block=True)
plt.close()
y_l, y_h = y_l-0.5, y_h+0.4
plt.figure()
plt.title('Argon Model ' + str(rho_arg) + ' ' + str(T_arg))
plt.xlabel('Radius (nm)')
plt.ylabel('NRLD')
plt.plot(exp_x, ynorm(exp_y, y_l, y_h), 'b')
plt.fill_between(exp_x, ynorm(exp_yl, y_l, y_h), ynorm(exp_yh, y_l, y_h), alpha=0.5, color='b',
                 edgecolor='w')
plt.plot(exp_x, ynorm(rdl, y_l, y_h), 'k')
plt.ylim((y_l, y_h))
plt.plot(exp_x, np.zeros(shape=len(exp_x)), 'k--')
plt.plot(exp_x, ynorm(models.rdf(exp_x, rho_arg, T_arg), y_l, y_h), 'r')
if save:
    out_name = ROOT_PATH + '/figs/nrld_argon_s' + str(sigma) + '_r' + str(rho_arg) + '_T' + \
               str(T_arg) + '.svg'
    plt.savefig(out_name, format='svg', dpi=1200)
else:
    plt.show(block=True)
plt.close()

print 'Terminated. (' + time.strftime("%c") + ')'

