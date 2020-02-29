from scipy.signal import chirp
import sounddevice as sd
import numpy as np
import time

A4b = 415.3
A4 = 440
A4s =  466.16

class Chirp:
    def __init__(self, tolerance, delay=0.1):
        self._delayInSeconds = delay
        self._tolerance = tolerance


    def _delay(self):
        if self._delayInSeconds > 0:
            time.sleep(self._delayInSeconds)

    @staticmethod
    def build_wave_from_channels(left_channel, right_channel, amplitude_modifier=1):
        int_16_multiplier = 32767
        modulation = amplitude_modifier * int_16_multiplier

        return np.int16(np.c_[left_channel, right_channel] * modulation)

    @staticmethod
    def get_monotone(tone=440.0, framerate=44100, duration_sec=1.0):
        t = np.linspace(0, duration_sec, int(duration_sec * framerate), endpoint=False)
        return chirp(t, f0=tone, t1=duration_sec, f1=tone, method='linear')

    @staticmethod
    def get_upwards_chirp(base_tone=440.0, framerate=44100, duration_sec=1.0):
        t = np.linspace(0, duration_sec, int(duration_sec * framerate), endpoint=False)
        return chirp(t, f0=base_tone, t1=duration_sec, f1=base_tone * 2, method='quadratic')

    @staticmethod
    def get_downwards_chirp(base_tone=440.0, framerate=44100, duration_sec=1.0):
        t = np.linspace(0, duration_sec, int(duration_sec * framerate), endpoint=False)
        return chirp(t, f0=base_tone, t1=duration_sec, f1=base_tone / 2, method='quadratic')

    def give_directions(self, target, centre):
        # Get target vector
        vector_hor = target[0] - centre[0] # Left = Negative
        vector_ver = target[1] - centre[1]

        left_amp = max(min(0.9, 0.5 - vector_hor), 0.1)
        right_amp = 1 - left_amp #max(min(0.9, 0.5 + vector_hor), 0.1)

        vertical_wave_tail = [] # pause for 0.3?
        if vector_ver > self._tolerance:
            vertical_wave_tail = Chirp.get_downwards_chirp(duration_sec=0.3)
        elif vector_ver < -self._tolerance:
            vertical_wave_tail = Chirp.get_upwards_chirp(duration_sec=0.3)

        monotone_head = Chirp.get_monotone(duration_sec=0.4)

        channel = np.r_[monotone_head, vertical_wave_tail]
        wave = Chirp.build_wave_from_channels(channel * left_amp, channel * right_amp, amplitude_modifier=1)

        with sd.OutputStream(dtype=np.int16, channels=2) as stream:
            stream.write(wave)

        self._delay()


if __name__ == '__main__':
    w1 = Chirp.get_monotone(tone=A4, duration_sec=0.4)
    w2 = Chirp.get_upwards_chirp(base_tone=A4, duration_sec=0.3)
    w3 = Chirp.get_downwards_chirp(base_tone=A4, duration_sec=0.3)

    channel = np.r_[w1, w3]
    w_int = Chirp.build_wave_from_channels(channel*0.5, channel*0.5, amplitude_modifier=1)

    with sd.OutputStream(dtype=np.int16, channels=2) as stream:
        stream.write(w_int)