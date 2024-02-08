# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:16:
print(spectra.shape)49 2024

@author: VK
"""

# Load data from .mat file
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

data = loadmat('data_sample.mat')
spectra = data['data_struct']
spectra=spectra['x']
spectra=spectra[0,0]
spectra=np.array(spectra)
spectra=np.transpose(spectra)
x=np.linspace(0,len(spectra),num=546)
plt.plot(spectra,x)
plt.show()