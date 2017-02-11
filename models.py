################################################################################################################
# Set of functions to model fluids particle distribution
#
#

__author__ = 'martinez'

import numpy as np

##### Global constants

JUMP_F = 0.

QR_ARR = (9.24792, -2.64281, 0.133386, -1.35932, 1.25338, 0.45602, -0.326422, 0.045708, -0.0287681,
          0.663161, -0.243089, 1.24749, -2.059, 0.04261, 1.65041, -0.343652, -0.037698, 0.008899,
          -0.0677912, -1.39505, 0.512625, 36.9323, -36.8061, 21.7353, -7.76671, 1.36342,
          16.4821, -0.300612, 0.0937844, -61.744, 145.285, -168.087, 98.2181, -23.0583,
          -8.33289, 2.1714, 1.0063,
          0.0325039, -1.28792, 2.5487,
          -26.1615, 27.4846, 1.68124, 6.74296,
          -6.7293, -59.5002, 10.2466, -0.43596,
          1.25225, -1.0179, 0.358564, -0.18533, 0.0482119, 1.27592, -1.78785, 0.634741,
          -5.668, -3.62671, 0.680654, 0.294481, 0.186395, -0.286954,
          6.01325, 3.84098, 0.60793
          )

##### Global functions

# For parsing and loading simulation's results
def read_simulation(fname):
    with open(fname) as file:
        file_s = file.readlines()
    arr = np.zeros(shape=(len(file_s), 2), dtype=float)
    for i, line in enumerate(file_s):
        # Finding delimiters
        ini, mid, en = line.find('('), line.find(','), line.find(')')
        # Formating
        arr[i, 0], arr[i, 1] = float(line[ini+1:mid]), float(line[mid+1:en])
    return arr

# Compute the squared error
def sqe_array(f, f_i):
    hold = f - f_i
    return hold * hold

# Compute the squared error mean between f and f_i
def sqe_mean(f, f_i):
    return np.mean(sqe_array(f, f_i))

# Compute the squared error max between f and f_i
def sqe_max(f, f_i):
    return np.max(sqe_array(f, f_i))

# Compute the squared error std between f and f_i
def sqe_std(f, f_i):
    return np.std(sqe_array(f, f_i))

# Normalizaton to convert from RDF to NRLD
def ynorm(y_i, y_l, y_h):
    y_o = np.zeros(shape=y_i.shape, dtype=y_i.dtype)
    mask = y_i >= 1
    y_o[mask] = y_i[mask] - 1.
    mask = np.invert(mask)
    y_o[mask] = -1. * ((1./y_i[mask]) - 1.)
    mask = y_o < y_l
    y_o[mask] = y_l
    mask = y_o > y_h
    y_o[mask] = y_h
    return y_o

###############################################################################################################
# Class with functions for RDF pure Lennard-Jones fluid models with physical parameters
# Morsali A. et al, J. Ghem. Phys., 310, 11-15, 2005
#
class LJ_Morsali(object):

    # sg: length parameter for LJ model
    # ref_x|ref_y: reference RDF values
    def __init__(self, sg, ref_x=None, ref_y=None):
        self.__sg = float(sg)
        if self.__sg <= 0:
            raise ValueError
        self.__ref_x, self.__ref_y = ref_x, ref_y
        self.__q_arr = np.asarray(QR_ARR, dtype=np.float)

    # r: interparticle radial distance (RDF X-axis)
    # rho: normalized density
    # T: normalized temperature
    def rdf(self, r, rho, T):
        a, b, c, d, g, h, k, l, m, n, s = self.functional_forms(rho, T)
        return self.__rdf(r, a, b, c, d, g, h, k, l, m, n, s)

    def sqe_rdf(self, params):
        return sqe_mean(self.rdf(self.__ref_x, params[0], params[1]), self.__ref_y)

    def functional_forms(self, rho, T):

        q_arr = self.__q_arr

        # Eq 2
        a = q_arr[0] + q_arr[1]*np.exp(-q_arr[2]*T) + q_arr[3]*np.exp(-q_arr[4]*T) + \
            q_arr[5]/rho + q_arr[6]/(rho**2) + \
            (q_arr[7]*np.exp(-q_arr[2]*T))/(rho**3) + \
            (q_arr[8]*np.exp(-q_arr[4]*T))/(rho**4)

        g = q_arr[9] + q_arr[10]*np.exp(-q_arr[11]*T) + q_arr[12]*np.exp(-q_arr[13]*T) + \
            q_arr[14]/rho + q_arr[15]/(rho**2) + \
            (q_arr[16]*np.exp(-q_arr[11]*T))/(rho**3) + \
            (q_arr[17]*np.exp(-q_arr[13]*T))/(rho**4)

        # Eq 3
        c = q_arr[18] + q_arr[19]*np.exp(-q_arr[20]*T) + q_arr[21]*rho + \
            + q_arr[22]*rho**2 + + q_arr[23]*rho**3 + + q_arr[24]*rho**4 + \
            + q_arr[25]*rho**5

        k = q_arr[26] + q_arr[27]*np.exp(-q_arr[28]*T) + q_arr[29]*rho + \
            + q_arr[30]*rho**2 + + q_arr[31]*rho**3 + + q_arr[32]*rho**4 + \
            + q_arr[33]*rho**5

        # Eq 4
        b = q_arr[34] + q_arr[35]*np.exp(-q_arr[36]*rho)

        h = q_arr[37] + q_arr[38]*np.exp(-q_arr[39]*rho)

        # Eq 5
        d = q_arr[40] + q_arr[41]*np.exp(-q_arr[42]*rho) + q_arr[43]*rho

        l = q_arr[44] + q_arr[45]*np.exp(-q_arr[46]*rho) + q_arr[47]*rho

        # Eq 6
        s = (q_arr[48] + q_arr[49]*rho + (q_arr[50]/T) + (q_arr[51]/T**2) + (q_arr[52]/T**3)) / \
            (q_arr[53] + q_arr[54]*rho + q_arr[55]*rho**2)

        # Eq 7
        m = q_arr[56] + q_arr[57]*np.exp(-q_arr[58]*T) + (q_arr[59]/T) + q_arr[60]*rho + \
            q_arr[61]*rho**2

        # Eq 8
        n = q_arr[62] + q_arr[63]*np.exp(-q_arr[64]*T)

        return  a, b, c, d, g, h, k, l, m, n, s

    ##### Internal functionality

    def __discont_jump(self, rho, T):

        a, b, c, d, g, h, k, l, m, n, s = self.functional_forms(rho, T)

        func_a = 1 + np.exp(-(a+b))*np.sin(c+d) + np.exp(-(g+h))*np.cos(k+l)
        func_b = s*np.exp(-((m+n)**4))
        err = func_a - func_b

        return err * err

    # x: input data 1D numpy array (valid range >= 0)
    # a, b, c, d, g, h, k, l, m, n, s: functional parameters
    def __rdf(self, x, a, b, c, d, g, h, k, l, m, n, s):

        y = np.zeros(shape=x.shape, dtype=np.float)
        x_n = x / self.__sg
        mask_1 = x_n > 1
        mask_2 = np.invert(mask_1)
        x_1, x_2 = x_n[mask_1], x_n[mask_2]

        term1 = np.exp(-(a*x_1+b)) * np.sin(c*x_1+d)
        term2 = np.exp(-(g*x_1+h)) * np.cos(k*x_1+l)
        y[mask_1] = 1. + np.power(x_1, -2.)*(term1 + term2)
        y[mask_2] = s * np.exp(-np.power(m*x_2+n, 4))

        return y
