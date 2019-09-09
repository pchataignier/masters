import speech_recognition as sr
import pyttsx3, time

class MicrophoneException(Exception): pass

#speech_recognition.RequestError -> API request error
#speech_recognition.UnknownValueError -> Can't recognize speech

class Listener:
    def __init__(self):
        self.Recognizer = sr.Recognizer()
        self.Mic = None
        self.StopEavesdropping = None
        self.keyword_found = False

    def GetMicrophone(self, micName=''):
        if not micName:
            self.Mic = sr.Microphone()
        else:
            mics = sr.Microphone.list_microphone_names()
            micFound = False
            for index,name in enumerate(mics):
                if micName in name:
                    micFound = True
                    break

            if not micFound:
                raise MicrophoneException('Unable to find named microphone')

            self.Mic = sr.Microphone(device_index=index)

        with self.Mic as source:
            self.Recognizer.adjust_for_ambient_noise(source)

    def Listen(self):
        with self.Mic as source:
            #self.Recognizer.adjust_for_ambient_noise(source)
            audio = self.Recognizer.listen(source) #phrase_time_limit:10, timeout:3 -> speech_recognition.WaitTimeoutError

        text = self.Recognizer.recognize_sphinx(audio)#, keyword_entries=[('something', 0.6)])
        return text

    def ListenForKeyword(self, keywords):
        #todo: checar se microphone existe
        print('Listening for: %s' % keywords)
        self.keyword_found = False
        search_terms = [(word, 0.25) for word in keywords]

        def keywordcallback(recognizer, audio):
            try:
                phrase = recognizer.recognize_sphinx(audio, keyword_entries=search_terms)
            except sr.UnknownValueError:
                phrase = None
            
            if phrase:
                print('Keyword found!')
                print(phrase)
                self.StopEavesdropping(wait_for_stop=False)
                self.keyword_found = True
            else:
                print('Not this time')

        #self.keyword_callback = keywordcallback
        self.StopEavesdropping = self.Recognizer.listen_in_background(self.Mic, keywordcallback)

        while not self.keyword_found:
            time.sleep(0.1)

    def ListenForCommand(self, grammar):
        with self.Mic as source:
            audio = self.Recognizer.listen(source) #phrase_time_limit:10, timeout:3 -> speech_recognition.WaitTimeoutError

        try:
            phrase = self.Recognizer.recognize_sphinx(audio, grammar=grammar)
        except sr.UnknownValueError:
            phrase = None
        
        return phrase
    
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
        

# Parrot main
if __name__ == '__main__':

    import os
    import re

    lis = Listener()
    spkr = Speaker()

    spkr.Queue("Hello human")
    spkr.Say("Give me a moment to get organized")

    pattern = r"(?P<find>(where)|(find))\s*(is)?\s*(the)?\s*(?P<thing>\w+)"

    try:
        lis.GetMicrophone('USB2.0')
    except MicrophoneException as e:
        print(str(e))
        spkr.Say(str(e))
        exit(1)

    spkr.Say("Ok, I'm ready")
    while True:
        lis.ListenForKeyword(['raspberry'])
        os.system('aplay -q ready.wav')

        grammar_file = 'pi_commands.fsg'
        text = lis.ListenForCommand(grammar_file)
        if not text:
            spkr.Say("Sorry, I didn't understand")
            continue

        print(text)
        match = re.match(pattern, text)
        if match:
            thing = match.group('thing')
        print(thing)
        spkr.Say("You are looking for the %s" % thing)


