import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.colors as mcolors

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import StftArgument

print('Reading audio file...')
data, sample_rate = sf.read("speech_sample.wav")
print(f'Audio data shape: {data.shape}, Sample rate: {sample_rate}')

print('Processing audio data...')
argument = StftArgument(t_int=0.02)
NT = int(argument.t_int*argument.sample_rate/2)
W_vector = np.array([ np.hanning(NT)[n] for n in np.arange(-NT, NT)])

Nchunks = len(data)//(2*NT)
data = data[:Nchunks*2*NT,0].reshape(Nchunks, 2*NT)*W_vector
print(f'Processed data shape: {data.shape}')

print('Computing Fmat and its eigensystem...')
Fmat = data.T @ data

#determine the eigensystem of Fmat and order the eigenvalues in decreasing order
eigenvalues, eigenvectors = np.linalg.eig(Fmat)
ii = np.argsort(-eigenvalues)
eigenvalues = eigenvalues[ii]
eigenvectors = eigenvectors[:,ii]
plt.plot(eigenvalues)
plt.yscale('log')
plt.xlabel('Index')
plt.ylabel('Eigenvalue (log scale)')
plt.title('Eigenvalues of Fmat')
plt.show()
