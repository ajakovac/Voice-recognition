import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.colors as mcolors

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import StftArgument


class LogBasis:
    def __init__(self, stft_argument):
        self.sample_rate     = stft_argument.sample_rate
        self.t_int           = stft_argument.t_int
        self.divide_step     = stft_argument.divide_step
        self.window_function = stft_argument.stft_default_window
        self.nu0 = 50 #Hz, the smallest frequency we consider
        self.Noctaves = 8 # number of octaves we consider
        self.Nmodes_per_octave = 24 # octave resolution
        self.Nalphas = self.Noctaves*self.Nmodes_per_octave
        self.t_step = self.t_int/self.divide_step
        self.dt = 1/self.sample_rate
        self.N_T = int(self.t_int*self.sample_rate)
        self.N_step = int(self.t_step*self.sample_rate)
        self.regulation = 1e-8

        self.nrange = np.arange(-self.N_T, self.N_T)
        self.W_vector = np.array([ self.window_function(n/self.N_T) for n in self.nrange])
        self.frequencies = np.array([ self.index_to_frequency(alpha) for alpha in range(self.Nalphas)])
        self.basis = np.array([ self.W_vector*np.exp(2j*np.pi*self.frequencies[n]*self.nrange*self.dt) for n in range(self.Nalphas) ]).T
        self.PCA_matrix = np.linalg.pinv(self.basis.T.dot(self.basis)+self.regulation*np.eye(self.Nalphas)).dot(self.basis.T)

    def index_to_frequency(self, alpha):
        return self.nu0*2**(alpha/self.Nmodes_per_octave)
    
    def frequency_to_index(self, nu):
        return int(self.Nmodes_per_octave*np.log2(nu/self.nu0))

    def transform(self, amplitude):
        embed_amplitude = np.array([ self.W_vector*amplitude[ncenter-self.N_T:ncenter+self.N_T] for ncenter in range(self.N_T, len(amplitude)-self.N_T, self.N_step) ])
        self.spectrogram = self.PCA_matrix.dot(embed_amplitude.T)
        self.window_times = np.arange(self.N_T, len(amplitude)-self.N_T, self.N_step)*self.dt
        return self.spectrogram, self.window_times, self.frequencies
