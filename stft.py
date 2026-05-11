import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import matplotlib.colors as mcolors

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import StftArgument

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

def play_sound(data, t1=0, t2=None, sample_rate=44100):
    if t2 is None:
        t2 = len(data)/sample_rate
    n1 = max(int(t1*sample_rate), 0)
    n2 = min(int(t2*sample_rate), len(data))
    data = data[n1:n2]
    channel_0_int16 = np.int16(data / np.max(np.abs(data)) * 32767)
    sd.play(channel_0_int16, samplerate=sample_rate)
    sd.wait()

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
        self.frequencies = np.fft.fftfreq(2*self.N_T, d=self.dt)[:self.N_T]
        self.clean()

    def clean(self):
        self.amplitude = None
        self.fourier_data = None
        return self

    def set_amplitude(self, amplitude):
        self.amplitude = amplitude
        self.fourier_data = None
        return self
    
    def set_fourier_data(self, fourier_data):
        self.amplitude = None
        self.fourier_data = fourier_data
        return self

    def transform(self):
        if self.amplitude is None:
            print('data is not set')
            return self
        self.fourier_data = []
        self.window_times = []
        for ncenter in range(self.N_T, len(self.amplitude)-self.N_T, self.N_step):
            fourier_trf = self.dt*np.fft.fft(self.W_vector*self.amplitude[ncenter-self.N_T:ncenter+self.N_T])[:self.N_T]
            self.fourier_data.append(fourier_trf)
            self.window_times.append(ncenter/self.sample_rate)
        self.fourier_data = np.array(self.fourier_data)
        self.window_times = np.array(self.window_times)
        return self
    
    def invert(self):
        if self.fourier_data is None:
            print('fourier data is not set')
            return self
        self.amplitude=np.array([])
        for data_part in self.fourier_data:
            ftrec = np.concatenate( (data_part, [0], np.conj(data_part[1:][::-1]) ), axis=0)
            new_rec =  np.real(np.fft.ifft(ftrec)/self.dt)
            if len(self.amplitude)==0:
                self.amplitude = new_rec
            else:
                self.amplitude = np.concatenate( ( self.amplitude, np.zeros(self.N_step) ))
                add_data = np.concatenate( ( np.zeros(len(self.amplitude) -len(new_rec) ), new_rec ))
                self.amplitude += add_data
        self.amplitude/=self.divide_step
        return self

    def slice(self, t1,t2):
        n1 = max(int(t1*self.sample_rate), 0)
        if t2 is None:
            t2 = len(self.amplitude)/self.sample_rate
        n2 = min(int(t2*self.sample_rate), len(self.amplitude))
        return self.amplitude[n1:n2]

    def plot_amplitude(self, t1=0, t2=None, ax = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        data = self.slice(t1, t2)
        subt = np.arange(len(data))/self.sample_rate
        ax.plot(subt, data, **kwargs)
        return fig, ax

    def plot_spectrum(self, t, line='-', ax = None, view='power', **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        n_time = int((t-self.window_times[0])/self.t_step)
        data = data_view(self.fourier_data[n_time], view)
        ax.plot(self.frequencies, data, line, **kwargs)
        ax.set_xscale('log')
        return fig, ax

    def plot_spectrogram(self, ax = None, view='spectrum', **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        alldata = data_view(self.fourier_data, view)[:,1:]
        colors = ["white", "blue"]  # White for negative, black for positive
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
        X, Y = np.meshgrid(self.window_times, self.frequencies[1:], indexing="ij")
        mesh = ax.pcolormesh(X,Y, alldata, shading='gouraud', cmap=cmap, **kwargs)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('frequency (Hz)')
        ax.set_yscale('log')
        return fig, ax,mesh

    def power(self):
        return np.sum(np.abs(self.fourier_data)**2,axis=1)

    def peaks(self, nmax=10, threshold=0, view='spectrum'):
        peaks = []
        for spectrum_data in data_view(self.fourier_data, view):
            peaks.append(find_maxima(spectrum_data, nmax=nmax, threshold=threshold))
        return peaks

def transform_to_logscale(spectrogram, nu0=50, Nmodes_per_octave=24, Noctaves=8):
    Nalphas = Nmodes_per_octave*Noctaves
    frequencies = np.array([ nu0*2**(alpha/Nmodes_per_octave) for alpha in range(Nalphas)])
    amplitude_transofrmation_matrix = np.zeros((Nalphas, spectrogram.fourier_data.shape[1]))
    for i, f in enumerate(frequencies):
        idx = np.argmin(np.abs(f-spectrogram.frequencies))
        amplitude_transofrmation_matrix[i, idx] = 1
    log_spectrogram = amplitude_transofrmation_matrix @ spectrogram.fourier_data.T
    return log_spectrogram.T, frequencies


class Synthesizer:
    def __init__(self, stft_argument:StftArgument=StftArgument()):
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
        self.frequencies = np.fft.fftfreq(2*self.N_T, d=self.dt)[:self.N_T]
        self.trange = np.arange(-self.N_T, self.N_T)/self.sample_rate

    def synthesize(self, chord_list):
        self.data = np.array([])
        tail = None
        for chord_data in chord_list:
            new_element = np.zeros_like(self.trange, dtype=complex)
            for amplitude, frequency in chord_data:
                sine_wave = amplitude*np.exp(2j*np.pi*frequency*self.trange)*self.W_vector
                if tail is not None:
                    head = sine_wave[:self.N_step]
                    phase_diff = np.angle(np.sum(tail*np.conj(head)))
                    sine_wave *= np.exp(1j*phase_diff)
                new_element += sine_wave
            if tail is not None:
                self.data = np.concatenate( ( self.data, np.zeros(self.N_step, dtype=complex) ))
                add_data = np.concatenate( ( np.zeros(len(self.data) -len(new_element), dtype=complex), new_element ))
                self.data += add_data
            else:
                self.data = new_element
            tail = new_element[-self.N_step:]
        self.data /= self.divide_step
        return self.data