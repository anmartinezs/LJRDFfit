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

################# Package import

import time
import numpy as np
from models import LJ_Morsali
import matplotlib.pyplot as plt

########################################################################################
# PARAMETERS
########################################################################################

ROOT_PATH = './'

###### Input of the experimental estimations

in_pkl_x = ROOT_PATH + '/data/mean_RLD_x.pkl'
in_pkl_y = ROOT_PATH + '/data/mean_RLD_y.pkl'
in_pkl_cil = ROOT_PATH + '/data/ci_99_low.pkl'
in_pkl_cih = ROOT_PATH + '/data/ci_99_high.pkl'

####### Lennar Jonnes for Argon

sigma = 13.9 # Length parameter of the LJ potential function
rho = 1.1
T = 5.1

# Range for r*
r_star = np.linspace(0, 5, 1000)

####### Showing results

save = True # If None the graphs are plotted in block mode
# Y-axis limits
y_l, y_h = 0, 3

########################################################################################
# MAIN ROUTINE
########################################################################################

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
print '\t\trho star: ' + str(rho)
print '\t\tT star: ' + str(T)
print '\tPlotting setting: '
if save:
    print '\t\t-Storing graphs directory: ' + ROOT_PATH + str('figs')
else:
    print '\t\t-Plotting graphs in block mode.'
print '\t\t-RDF Y-axis limits: ' + str((y_l, y_h))
print ''

######### Process

print 'Main Routine: '

print '\tInitializing the models...'
model = LJ_Morsali(sigma)

print '\tComputing RDF...'
rdf = model.rdf(r_star*sigma, rho, T)

print '\tPlotting results...'
plt.figure()
plt.title('Morsali model for Argon')
plt.xlabel('Radius (nm)')
plt.ylabel('RLD')
plt.plot(r_star, rdf, 'b')
if save:
    out_name = ROOT_PATH + '/figs/rdf_argon_s' + str(sigma) + '_r' + str(rho) + '_T' + \
               str(T) + '.png'
    plt.savefig(out_name, format='png', dpi=1200)
else:
    plt.show(block=True)
plt.close()

print 'Terminated. (' + time.strftime("%c") + ')'

