import pyttsx3, time

class Speaker:
    def __init__(self, tolerance, delay=0.1):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 130)
        #todo: definir por um dict
        self._delayInSeconds = delay
        self._tolerance = tolerance

        # Patch for Windows voices TODO: do this right
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if "English" in voice.name:
                self.engine.setProperty('voice', voice.id)
                break

    def _delay(self):
        if self._delayInSeconds > 0:
            time.sleep(self._delayInSeconds)

    def say(self, phrase):
        self.engine.say(phrase)
        self.engine.runAndWait()
        self._delay()

    def queue(self, phrase):
        self.engine.say(phrase)

    def flush(self):
        self.engine.runAndWait()
        self._delay()

    def give_directions(self, target, centre):
        # Get target vector
        vector_hor = target[0] - centre[0]
        vector_ver = target[1] - centre[1]

        hor_centered = False
        ver_centered = False

        # Queue up directions
        if vector_hor > self._tolerance:
            self.queue("Right")
        elif vector_hor < -self._tolerance:
            self.queue("Left")
        else:
            hor_centered = True

        if vector_ver > self._tolerance:
            self.queue("Down")
        elif vector_ver < -self._tolerance:
            self.queue("Up")
        else:
            ver_centered = True

        if hor_centered and ver_centered:
            self.queue("Ahead")

        # Give directions
        self.flush()
        self._delay()