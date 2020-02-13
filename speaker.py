import pyttsx3, time

class Speaker:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 130)
        #todo: definir por um dict

    def Say(self, phrase):
        self.engine.say(phrase)
        self.engine.runAndWait()
        time.sleep(0.1)

    def Queue(self, phrase):
        self.engine.say(phrase)

    def Flush(self):
        self.engine.runAndWait()
        time.sleep(0.1)