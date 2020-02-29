import math
import time
import numpy as np
import sounddevice as sd

# A4b = 415.3
# A4 = 440
# A4s =  466.16

class Beeper:
    def __init__(self, tolerance, delay=0.1):
        self._delayInSeconds = delay
        self._tolerance = tolerance

        self._tone_down = 415.3
        self._tone_centre = 440.0
        self._tone_up = 466.16

    def _delay(self):
        if self._delayInSeconds > 0:
            time.sleep(self._delayInSeconds)


    @staticmethod
    def sine_wave(frequency=440.0, framerate=44100, amplitude=0.5, duration_sec=1.0):
        period = int(framerate / frequency)
        if amplitude > 1.0: amplitude = 1.0
        if amplitude < 0.0: amplitude = 0.0
        lookup_table = [float(amplitude) * math.sin(2.0*math.pi*float(frequency)*(float(i%period)/float(framerate))) for i in range(period)]
        n_samples = int(framerate * duration_sec)
        return np.array([lookup_table[i%period] for i in range(n_samples)])


    @staticmethod
    def pause_wave(framerate=44100, duration_sec=1.0):
        n_samples = int(framerate * duration_sec)
        return np.zeros(n_samples, "int16")


    @staticmethod
    def build_wave_from_channels(left_channel, right_channel):
        int_16_multiplier = 32767
        amplitude_modifier = 1
        modulation = amplitude_modifier * int_16_multiplier

        return np.int16(np.c_[left_channel, right_channel] * modulation)


    @staticmethod
    def get_single_beep(tone, left_amplitude, right_amplitude):
        samples_left = Beeper.sine_wave(frequency=tone, amplitude=left_amplitude, duration_sec=0.4)
        samples_right = Beeper.sine_wave(frequency=tone, amplitude=right_amplitude, duration_sec=0.4)
        pause = Beeper.pause_wave(duration_sec=0.1)

        wave_left = np.concatenate([pause, samples_left, pause])
        wave_right = np.concatenate([pause, samples_right, pause])

        return Beeper.build_wave_from_channels(wave_left, wave_right)


    @staticmethod
    def get_double_beep(tone, left_amplitude, right_amplitude):
        samples_left = Beeper.sine_wave(frequency=tone, amplitude=left_amplitude, duration_sec=0.3)
        samples_right = Beeper.sine_wave(frequency=tone, amplitude=right_amplitude, duration_sec=0.3)
        pause = Beeper.pause_wave(duration_sec=0.1)

        wave_left = np.concatenate([pause, samples_left, pause, samples_left, pause])
        wave_right = np.concatenate([pause, samples_right, pause, samples_right, pause])

        return Beeper.build_wave_from_channels(wave_left, wave_right)


    @staticmethod
    def get_triple_beep(tone, left_amplitude, right_amplitude):
        samples_left = Beeper.sine_wave(frequency=tone, amplitude=left_amplitude, duration_sec=0.3)
        samples_right = Beeper.sine_wave(frequency=tone, amplitude=right_amplitude, duration_sec=0.3)
        pause = Beeper.pause_wave(duration_sec=0.1)

        wave_left = np.concatenate([pause, samples_left, pause, samples_left, pause, samples_left, pause])
        wave_right = np.concatenate([pause, samples_right, pause, samples_right, pause, samples_right, pause])

        return Beeper.build_wave_from_channels(wave_left, wave_right)


    def give_directions(self, target, centre):
        # Get target vector
        vector_hor = target[0] - centre[0] # Left = Negative
        vector_ver = target[1] - centre[1]

        left_amp = max(min(0.9, 0.5 - vector_hor), 0.1)
        right_amp = 1 - left_amp #max(min(0.9, 0.5 + vector_hor), 0.1)

        tone = self._tone_centre
        if vector_ver > self._tolerance:
            tone = self._tone_down
        elif vector_ver < -self._tolerance:
            tone = self._tone_up

        wave = self.get_single_beep(tone, left_amplitude=left_amp, right_amplitude=right_amp)
        with sd.OutputStream(dtype=np.int16, channels=2) as stream:
            stream.write(wave)

        self._delay()


# stream = sd.OutputStream(dtype=np.int16, channels=2)
# stream.start()
# # time.sleep(0.2)
#
# for tone in [A4, A4b, A4, A4s]:
#     # samples_left = sine_wave(frequency=tone, amplitude=0.5, duration_sec=0.3)
#     # samples_right = sine_wave(frequency=tone, amplitude=0.5, duration_sec=0.3)
#     # pause = pause_wave(duration_sec=0.1)
#     #
#     # wave_left = np.concatenate([pause, samples_left, pause, samples_left, pause, samples_left])
#     # wave_right = np.concatenate([pause, samples_right, pause, samples_right, pause, samples_right])
#     #
#     # int_16_multiplier = 32767
#     # amplitude_modifier = 1
#     # modulation = amplitude_modifier * int_16_multiplier
#     #
#     # w_int = np.int16(np.c_[wave_left, wave_right] * modulation)
#     beeper = Beeper(0.2)
#     w_int = beeper.get_single_beep(tone, left_amplitude=0.5, right_amplitude=0.5)
#     print(w_int.shape[0]/44100)
#
#     # sd.play(w_int, 44100, blocking=True)
#     # with sd.OutputStream(dtype=np.int16, channels=2) as stream:
#     stream.write(w_int)
#     # time.sleep(0.5)
#
# stream.stop()
# stream.close()