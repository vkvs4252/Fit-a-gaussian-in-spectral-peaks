# Load data from .mat file
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import chirp, find_peaks, peak_widths
import numpy as np

spectra = loadmat('denoised_exp.mat')
spectra=spectra['denoised_exp']
spectra=np.squeeze(spectra, axis=0)
#spectra=np.narray(spectra)
'''spectra = data['data_struct']
spectra=spectra['x']
spectra=spectra[0,0]
spectra=np.array(spectra)
spectra=spectra.transpose()'''
peaks, _ = find_peaks(spectra)
results_half = peak_widths(spectra, peaks, rel_height=0.5)
#results_full = peak_widths(spectra, peaks, rel_height=1)

plt.plot(spectra)
#plt.plot(peaks, spectra[peaks], "x")
plt.hlines(*results_half[1:], color="C2")

#plt.hlines(*results_full[1:], color="C3")
plt.show()
