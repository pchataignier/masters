#!/usr/bin/python3

from scipy.signal import chirp
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os, time

STEREO_LEFT = -1
STEREO_CENTER = 0
STEREO_RIGHT = 1

VERTICAL_UPWARDS = 1
VERTICAL_CONSTANT = 0
VERTICAL_DOWNWARDS = -1

class Chirper:
    def __init__(self):
        self.sampling_rate = 44100
        self.int_16_multiplier = 32767
        self.harmonic_modifier = 3
        self.base_frequency = 1100

    def get_chirp(self, duration, vertical_dir=0, stereo_side=0, amplitude_modifier=0.5):
        t = np.linspace(0, duration, duration * self.sampling_rate, endpoint=False)
        f0 = self.base_frequency
        f1=f0
        if vertical_dir > 0:
            f1 = f0 * self.harmonic_modifier
        elif vertical_dir < 0:
            f1 = f0/self.harmonic_modifier

        w = chirp(t, f0, duration, f1, method='linear')

        modulation = amplitude_modifier * self.int_16_multiplier
        w_int = np.int16(w * modulation)

        if stereo_side == 0:
            return w_int

        z = np.zeros(w_int.size, dtype=np.int16)
        #z = np.int16(w_int*(1-stereo_side))
        if stereo_side > 0: # right
            w_int = np.c_[z, w_int]
        elif stereo_side < 0: # left
            w_int = np.c_[w_int, z]

        return w_int


    def play_chirp(self, wave):
        sd.play(wave, self.sampling_rate, blocking=True)
        #write('test.wav', self.sampling_rate, wave)
        #time.sleep(0.2)
        #os.system('aplay -q test.wav')

    def write_wav(self, wave, filename):
        if not filename.endswith('.wav'):
            filename += '.wav'
        write(filename, self.sampling_rate, wave)

if __name__ == '__main__':
    chirper = Chirper()
    T = 0.25
    #chirper.base_frequency=440

    #start = time.perf_counter()
    w = chirper.get_chirp(T, vertical_dir=VERTICAL_UPWARDS, stereo_side=STEREO_LEFT, amplitude_modifier=0.7)
    #end = time.perf_counter()
    print(end - start)

    #start = time.perf_counter()
    chirper.play_chirp(w)
    #end = time.perf_counter()
    print(end - start)
