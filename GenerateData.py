import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from numpy import random
from numpy import zeros
from numpy.random import randint
from numpy.random import uniform
from numpy.random import normal
from numpy import array
from numpy import sqrt
from numpy import pi
import time
import os

def G(x, xo, alpha):
    """ Return Gaussian line shape at x with HWHM alpha """
    return sqrt(np.log(2) / pi) / alpha \
        * np.exp(-((x - xo) / alpha) ** 2 * np.log(2))
def L(x, xo, gammao):
    """ Return Lorentzian line shape at x with HWHM gammao """
    return gammao / pi / ((x - xo) ** 2 + gammao ** 2)
def Select_Random_Peaks(peak_number_range, peak_sd_range):
    no = randint(peak_number_range[0], peak_number_range[1])  # no - number of peaks
    xo = uniform(0, x_max, no)  ## list of Lorentzian peak x-position ,don't need to be at select linspace point
    gammao = uniform(peak_sd_range[0], peak_sd_range[1], no)  ## list of Lorentzian peak s.d.
    ho = uniform(size=no)  ## list of Lorentzian height :  assume between 0 to 1
    return no, xo, gammao, ho


def Produce_G_Peaks(x, xo, alpha, ho):
    y = array(G(x, xo,
                alpha) * ho * alpha)  ### final alpha[i] to normalise wider slit and narrower slit , to make them have same height
    return y


def Produce_L_Peaks(x, xo, gammao, h, N_points):
    y = zeros(N_points)
    for i in range(np.size(xo)):
        yn = array(L(x, xo[i], gammao[i]) * h[i] * gammao[i])
        y = y + yn
    return y


def rescale(y):
    y = y * 5000
    return y


def add_noise(y):
    sigma_shot_noise = sqrt(y)
    sigma_readout = uniform(0, 200, len(y))
    sigma_dark_current = uniform(0, 100, len(y))
    sigma_total_noise = sqrt(sigma_readout ** 2 + sigma_dark_current ** 2 + sigma_shot_noise ** 2)
    y_input = normal(y, sigma_total_noise, len(y))
    return y_input


def normalisation(y):
    y = y / max(y)
    return y


starttime=time.time()
# 1.Parameters set ()
N_sets = 10000
Data_folder='C:/Users/VK/Desktop/PythonScripts/Data/'
#   Create x axis
x_min = -160
x_max = 2840
N_points = 1500
NN_points = 1340
dx = 2
x = np.arange(x_min, x_max, dx)


#   creat saving space as matrixe of Y_TARGET/Y_INPUT
Y_TARGET = zeros((N_sets, NN_points))
Y_INPUT = zeros((N_sets, NN_points))

# 2. Generate spectra database
for iter in range(10):
    for j in range(N_sets):
        #print(j)
        #   Lorentzian - Random_setting(peaks number range, peak s.d. range
        no, xo, gammao, ho = Select_Random_Peaks([15, 30], [10, 40])

        #  Lorentzian - produce spectra
        y_L1 = Produce_L_Peaks(x, xo, gammao, ho,N_points)
        y_L = normalisation(y_L1)

        #  Gaussain - Prepare for convolution
        alpha_1 = uniform(0.02, 0.02) * x_max
        alpha_2 = uniform(0.001, 0.001) * x_max
        #  Gaussian - height
        ho_G = 0.1
        #  create saving space for convolution result
        y_conv = zeros(N_points)
        y_conv_2 = zeros(N_points)

        #  Convolution to produce Voigt
        for i in range(N_points):

            y_G_1 = Produce_G_Peaks(x, x[i], alpha_1, ho_G)
            y_G_1_max = max(y_G_1)
            y_convv=y_G_1 * y_L * dx
            y_conv[i] = sum(y_convv)

            y_G_2 = Produce_G_Peaks(x, x[i], alpha_2, ho_G)
            y_G_2_max = max(y_G_2)
            y_convv_2=y_G_2 * y_L * dx
            y_conv_2[i] = sum(y_convv_2)

        y_input = y_conv [80:1420]
        y_target = y_conv_2 [80:1420]
        y_L = y_L[80:1420]

        #  rescale to make noise reasonable
        y_input = rescale(y_input)
        y_target = rescale(y_target)

        #  Add noise
        y_input = add_noise(y_input)

        #  Normalisation
        y_target_n = normalisation(y_target)
        y_input_n = normalisation(y_input)



        Y_TARGET[j] = y_target_n  # create a matrix for dataset
        Y_INPUT[j] = y_input_n  # create a matrix for dataset

    print('production of next '+ str(N_sets)+ ' datasets takes %s seconds', time.time()-starttime)
        #  Save
    io.savemat(Data_folder + '240121_target' + str(N_sets) + '_' + str(iter)+'.mat', {"data": Y_TARGET})
    io.savemat(Data_folder + '240121_input' + str(N_sets) + '_' + str(iter)+'.mat', {"data": Y_INPUT})

'''   
#  Plot ax1(raw) , ax2(Normalised)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)


#  ax1
ax1.plot(x[80:1420], y_input, label='Natural + wide slit', color='red')
ax1.plot(x[80:1420], y_target, label='Natural + narrow slit', color='black')
ax1.plot(x[80:1420], y_L1[80:1420], label ='Natural', color='black')
ax1.set_title('Real value(1340 points)')
ax1.legend()

#  ax2
ax2.plot(x[80:1420], y_input_n, label='Natural + wide slit ', color='red')
ax2.plot(x[80:1420], y_target_n, label='Natural + narrow slit', color='black')
ax2.plot(x[80:1420], y_L, label ='Natural', color='green')
ax2.set_title('Normalisation (1340 points)')
ax2.legend()

plt.show()

'''
