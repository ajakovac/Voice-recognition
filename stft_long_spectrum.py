import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.colors as mcolors

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import stft_argument

def find_maxima(data, xrange=None, nmax=None, threshold=None, interpolate=True):
    if xrange is None:
        xrange=np.arange(len(data))
    T = np.array([xrange**2, xrange, np.ones(xrange.shape)]).T
    ddata = data[1:]-data[:-1]
    max_indices = np.arange(len(ddata)-1)[np.logical_and(ddata[1:]*ddata[:-1]<=0,ddata[:-1]>0)]+1
    if len(max_indices)==0:
        return [],[]
    max_values = data[max_indices]

    if interpolate:
        Islice = np.array([max_indices-1, max_indices, max_indices+1]).T
        coeff =np.array([np.linalg.inv(T[Ii])@data[Ii] for Ii in Islice])
        max_positions = -coeff[:,1]/(2*coeff[:,0])
        max_values = np.diag(coeff@np.array([max_positions**2, max_positions, np.ones(len(max_positions))]))
    else:
        max_positions = xrange[max_indices]

    if threshold is not None:
        max_positions = max_positions[max_values>threshold]
        max_values = max_values[max_values>threshold]
    ii = np.argsort(-max_values)
    if nmax is not None:
        ii = ii[:nmax]
    return max_positions[ii],max_values[ii]

def data_view(data, view):
    try:
        if view == 'power':
            return np.abs(data)**2
        if view == 'spectrum':
            return np.abs(data)
        elif view == 'phases':
            return np.angle(data)
        elif view == 'real':
            return np.real(data)
        elif view == 'imag':
            return np.imag(data)
        else:
            print(f'view {view} is not supported')
            return None
    except Exception as e:
        print(f'Error in view function: {e}')
        return None

class SoundData:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.sampling_dt = 1/sample_rate
    
    def set(self, data):
        self.data = data
        self.duration = len(data)/self.sample_rate  # maximal time of the signal
        return self

    def time_to_index(self, ti):
        if ti is None:
            return int(self.duration*self.sample_rate)
        elif ti>self.duration:
            print(f'time {ti} is out of range: it must be in [0,{self.duration}]')
            return int(self.duration*self.sample_rate)
        elif ti< 0:
            print(f'time {ti} is out of range: it must be in [0,{self.duration}]')
            return 0
        else:
            return int(ti*self.sample_rate)

    def slice(self, t1,t2):
        n1 = self.time_to_index(t1)
        n2 = self.time_to_index(t2)
        return self.data[n1:n2]

    def play(self, t1=0, t2=None):
        channel_0_int16 = np.int16(self.slice(t1,t2) / np.max(np.abs(self.slice(t1,t2))) * 32767)
        sd.play(channel_0_int16, samplerate=self.sample_rate)
        sd.wait()

    def plot(self, ax = None, t1=0, t2=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        data = self.slice(t1, t2)
        subt = np.arange(len(data))/self.sample_rate
        ax.plot(subt, data, **kwargs)
        return fig, ax

class STFT:
    def __init__(self, stft_argument):
        self.sample_rate     = stft_argument.sample_rate
        self.t_int           = stft_argument.t_int
        self.divide_step     = stft_argument.divide_step
        self.window_function = stft_argument.stft_default_window
        
        self.t_step = self.t_int/self.divide_step
        self.dt = 1/self.sample_rate
        self.N_T = int(self.t_int*self.sample_rate)
        self.N_step = int(self.t_step*self.sample_rate)
        xrange = np.arange(-self.N_T, self.N_T)
        self.W_vector = np.array([ self.window_function(n/self.N_T) for n in xrange])
        self.norm = np.sum(self.W_vector**2)/len(xrange)
        self.frequencies = np.fft.fftfreq(2*self.N_T, d=self.dt)[:self.N_T+1]
        self.frequencies[-1] *= -1

    def range(self, t0, t1):
        return np.arange(t0,t1+self.dt, self.dt)

    def __call__(self, data):
        self.times = []
        self.data = []
        sign_vector = np.array([(-1)**n for n in range(len(self.W_vector))])
        for ncenter in range(self.N_T, len(data)-self.N_T, self.N_step):
            self.times.append(ncenter/self.sample_rate)
            fourier_data = np.fft.fft(self.W_vector*data[ncenter-self.N_T:ncenter+self.N_T])[:self.N_T+1]
            self.data.append(self.dt*fourier_data)
        self.times = np.array(self.times)
        self.data = np.array(self.data)
        return self

    def plot_spectrum(self, t, line='-', ax = None, view='power', **kwargs):
        n_time = int((t-self.times[0])/self.t_step)
        data = data_view(self.data[n_time], view)
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.frequencies, data, line, **kwargs)
        ax.set_xscale('log')
        return fig, ax

    def plot(self, ax = None, view='spectrum', **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        alldata = data_view(self.data, view)[:,1:]
        colors = ["white", "blue"]  # White for negative, black for positive
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        X, Y = np.meshgrid(self.times, self.frequencies[1:], indexing="ij")
        mesh = ax.pcolormesh(X,Y, alldata, shading='gouraud', cmap=cmap, **kwargs)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('frequency (Hz)')
        ax.set_yscale('log')
        return fig, ax,mesh

    def power(self):
        return np.sum(np.abs(self.data)**2,axis=1)

    def peaks(self, nmax=10, threshold=0, view='spectrum'):
        data = data_view(self.data, view)
        peaks = []
        for spectrum_data in data:
            peaks.append(find_maxima(spectrum_data, nmax, threshold))
        return peaks

class Synthetize:
    def __init__(self, stft_argument):
        self.sample_rate     = stft_argument.sample_rate
        self.t_int           = stft_argument.t_int
        self.divide_step     = stft_argument.divide_step
        self.window_function = stft_argument.stft_default_window
        
        self.t_step = self.t_int/self.divide_step
        self.dt = 1/self.sample_rate
        self.N_T = int(self.t_int*self.sample_rate)
        self.N_step = int(self.t_step*self.sample_rate)
        self.sign_vector = np.array([(-1)**n for n in range(2*self.N_T)])
        xrange = np.arange(-self.N_T, self.N_T)
        self.W_vector = np.array([ self.window_function(n/self.N_T) for n in xrange])
        self.norm = np.sum(self.W_vector**2)/len(xrange)
        self.frequencies = np.fft.fftfreq(2*self.N_T, d=self.dt)[:self.N_T+1]
        self.frequencies[-1] *= -1

    def set_data(self, spectrogram):
        self.data = spectrogram
        return self

    def synthetize_slice(self, data_part):
        ftrec = np.concatenate( (data_part, np.conj(data_part[1:-1][::-1]) ), axis=0)
        return np.real(np.fft.ifft(ftrec)/self.dt)
    
    def synthetize(self):
        rec_data=np.array([])
        for data_part in self.data:
            new_rec = self.synthetize_slice(data_part)
            if len(rec_data)==0:
                rec_data = new_rec
            else:
                rec_data = np.concatenate( ( rec_data, np.zeros(self.N_step) ))
                add_data = np.concatenate( ( np.zeros(len(rec_data)-2*self.N_T), new_rec ))
                rec_data += add_data
        return rec_data/self.divide_step