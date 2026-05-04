import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import StftArgument

@dataclass
class NoteBasis:
    ratio: float = 2 ** (1/12)
    A4: float = 440 #Hz

NoteNames = { 
    'C' : NoteBasis.A4 * NoteBasis.ratio**(-9),
    'C#': NoteBasis.A4 * NoteBasis.ratio**(-8),
    'D' : NoteBasis.A4 * NoteBasis.ratio**(-7),
    'D#': NoteBasis.A4 * NoteBasis.ratio**(-6),
    'E' : NoteBasis.A4 * NoteBasis.ratio**(-5),
    'F' : NoteBasis.A4 * NoteBasis.ratio**(-4),
    'F#': NoteBasis.A4 * NoteBasis.ratio**(-3),
    'G' : NoteBasis.A4 * NoteBasis.ratio**(-2),
    'G#': NoteBasis.A4 * NoteBasis.ratio**(-1),
    'A' : NoteBasis.A4,
    'A#': NoteBasis.A4 * NoteBasis.ratio,
    'B' : NoteBasis.A4 * NoteBasis.ratio**2,
    'c' : NoteBasis.A4 * NoteBasis.ratio**3,
    'c#': NoteBasis.A4 * NoteBasis.ratio**4,
    'd' : NoteBasis.A4 * NoteBasis.ratio**5,
    'd#': NoteBasis.A4 * NoteBasis.ratio**6,
    'e' : NoteBasis.A4 * NoteBasis.ratio**7,
    'f' : NoteBasis.A4 * NoteBasis.ratio**8,
    'f#': NoteBasis.A4 * NoteBasis.ratio**9,
    'g' : NoteBasis.A4 * NoteBasis.ratio**10,
    'g#': NoteBasis.A4 * NoteBasis.ratio**11,
    'a' : NoteBasis.A4 * NoteBasis.ratio**12,
    'a#': NoteBasis.A4 * NoteBasis.ratio**13,
    'b' : NoteBasis.A4 * NoteBasis.ratio**14,
    '' : 0
}

@dataclass
class Note:
    length: float
    note: str
    pitch: int = 0
    A: float = 1.0

def note_from_string(s):
    parts = s.split(':')
    note = parts[0]
    length = float(parts[1])
    pitch_str = parts[2] if len(parts) > 2 else 0
    if pitch_str == '':
        pitch = 0
    else:
        pitch = int(pitch_str)
    A = float(parts[3]) if len(parts) > 3 else 1.0
    return Note(length, note, pitch, A)

@dataclass
class FluteShaper:
    sample_rate: int = 44100
    dt: float = 1/sample_rate
    attack_time: float = 0.08
    peak_height: float = 1.1
    decay_time: float = 0.02
    release_time: float = 0.2
    attack_mask: np.ndarray = field(init=False)
    decay_mask: np.ndarray = field(init=False)
    release_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        t = np.arange(0, self.attack_time, self.dt)
        self.attack_mask = t / self.attack_time * self.peak_height
        t = np.arange(0, self.decay_time, self.dt)
        self.decay_mask = (1 - t / self.decay_time) * (self.peak_height - 1) + 1
        t = np.arange(0, self.release_time, self.dt)
        self.release_mask = (1 - t / self.release_time)
    
    @staticmethod
    def envelope(length):
        if length < FluteShaper.attack_time + FluteShaper.decay_time + FluteShaper.release_time:
            raise ValueError('Note is too short for the envelope')
        attack_length = int(FluteShaper.attack_time / FluteShaper.dt)
        decay_length = int(FluteShaper.decay_time / FluteShaper.dt)
        release_length = int(FluteShaper.release_time / FluteShaper.dt)
        sustain_length = int(length / FluteShaper.dt) - attack_length - decay_length - release_length
        if sustain_length < 0:
            raise ValueError('Note is too short for the envelope')
        attack_vector = np.linspace(0, FluteShaper.peak_height, attack_length)
        decay_vector = np.linspace(FluteShaper.peak_height, 1, decay_length)
        sustain_vector = np.ones(sustain_length)
        release_vector = np.linspace(1, 0, release_length)
        return np.concatenate([attack_vector, decay_vector, sustain_vector, release_vector])

class Synthetizer:
    def __init__(self, time_unit=1.0, stft_argument:StftArgument=StftArgument()):
        self.time_unit = time_unit
        self.sample_rate     = stft_argument.sample_rate
        self.dt = 1/self.sample_rate
        self.t_int           = stft_argument.t_int
        self.divide_step     = stft_argument.divide_step
        self.N_T = int(self.t_int * self.sample_rate)
        xrange = np.arange(-self.N_T, self.N_T)
        window_function = stft_argument.wavelet_connector_factory()
        self.W_vector = np.array([ window_function(n/self.N_T) for n in xrange])

        self.A4 = 440 #Hz
        ratio = 2 ** (1/12)

        self.channels = {}

    def make_sound(self, note:Note|str):
        if isinstance(note, str):
            note = Note.from_string(note)
        nu = self.notes[note.note]*(2**note.pitch)
        if nu == 0:
            return np.zeros(int(self.time_unit * float(note.length) / self.dt)), 0
        compensated_amplitude = note.A * self.A4/nu
        trange = np.arange(0, self.time_unit * float(note.length), self.dt)
        data = compensated_amplitude * np.sin(2 * np.pi * nu * trange)
        dt_window = self.dt * 2/note.fade
        window_vector = self.window(dt_window)
        wlength = window_vector.shape[0]
        data_length = data.shape[0]
        length_to_add = data_length - wlength
        if length_to_add < 0:
            raise ValueError('Note is too short for the fade time')
            #mask = np.concatenate([window_vector[:data_length//2], window_vector[wlength//2:]])
        else:
            mask = np.concatenate([window_vector[:wlength//2], np.ones(length_to_add), window_vector[wlength//2:]])
        return mask*data, wlength//2
    
    def clear_channel(self, channel='default'):
        self.channels[channel] = {
            'fade_time': 0.1,
            'fade_function': lambda x, ftime: (np.cos(np.pi * x/ftime) +1)/2,
            'data': np.array([])
        }

    def add_sound(self, note:Note|str, channel='default'):
        if isinstance(note, str):
            note = Note.from_string(note)
        addon,NT = self.make_sound(note)
        d1 = np.concatenate([self.channels[channel]['data'], np.zeros(addon.shape[0]-NT)])
        if self.channels[channel]['data'].shape[0] > NT:
            d2 = np.concatenate([np.zeros(self.channels[channel]['data'].shape[0]-NT), addon])
            self.channels[channel]['data'] = d1 + d2
        else:
            self.data[channel] = addon

    def combine_channels(self, channel1, channel2, new_channel='combined'):
        d1 = self.data[channel1]
        d2 = self.data[channel2]
        if d1.shape[0] > d2.shape[0]:
            d2 = np.concatenate([d2, np.zeros(d1.shape[0]-d2.shape[0])])
        else:
            d1 = np.concatenate([d1, np.zeros(d2.shape[0]-d1.shape[0])])
        self.data[new_channel] = d1 + d2

    def play(self, channel='combined'):
        if channel not in self.data:
            channel = 'default'
        channel_0_int16 = np.int16(self.data[channel] / np.max(np.abs(self.data[channel])) * 32767)
        sd.play(channel_0_int16, samplerate=self.sample_rate)
        sd.wait()

    def save(self, filename, channels=['default']):
        data = np.stack([self.data[channel] for channel in channels], axis=1)
        sf.write(filename, data, self.sample_rate)
