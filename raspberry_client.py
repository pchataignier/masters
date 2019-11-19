#!/usr/bin/python3
from socketUtils import *
from imutils.video import VideoStream
from parrot import *
import argparse
import re
import time
import socket
import pickle
import uuid
import os

CLASSES = {'telephone':1, 'mug':2, 'remote control':3, 'remote':3, 'bottle':4, 'hand':5}
class_id = "class_id"
mask = "mask"
roi = "roi"
score = "score"

class InferenceClient:
    def __init__(self, host, port):
        self.Host = host
        self.Port = port
        self.Server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Server.connect((host, port))

    def GetDetections(self, img_id, frame):
        data = EvaluationData(img_id, frame)

        pkg = pickle.dumps(data)
        send_msg(self.Server, pkg)

        data = recv_msg(self.Server)
        resp = pickle.loads(data)
        return resp

    def Close(self):
        self.Server.close()

def ip_with_port(arg):
    if re.match(r'\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3}[:]\d+', arg):
        return arg
    raise argparse.ArgumentTypeError('Server address must be follow the format \'0.0.0.0:<port>\'')

def play_ready():
    os.system('aplay -q ready.wav')

def play_error():
    os.system('aplay -q error.wav')

def get_detection_centre(detection):
    return ( ((detection[roi][3] - detection[roi][1]) / 2) + detection[roi][1], ((detection[roi][2] - detection[roi][0]) / 2) + detection[roi][0] )

def get_detection_area(detection):
    return (detection[roi][3] - detection[roi][1]) * (detection[roi][2] - detection[roi][0])

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--server", required=True, type=ip_with_port, help="IP and port of the server being used for detection. Required format - 0.0.0.0:<port>")
parser.add_argument("-P", "--picamera", action="store_true", help="Whether or not the Raspberry Pi camera should be used")

args = parser.parse_args()

host, port = args.server.split(':')
port = int(port)
client = InferenceClient(host, port)

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args.picamera).start()
time.sleep(2.0)

###
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
    play_ready()

    grammar_file = 'pi_commands.fsg'
    text = lis.ListenForCommand(grammar_file)
    if not text:
        spkr.Say("Sorry, I didn't understand")
        continue

    print(text)
    match = re.match(pattern, text)
    if not match:
        play_error()
        continue

    thing = match.group('thing')
    print(thing)
    spkr.Say("You are looking for the %s" % thing)

    targetClass = CLASSES[thing]
    # Guidance loop
    while True: # TODO: stop when found
        # Get frame and run it through inferece
        frame = vs.read()
        frame_size = frame.shape
        img_id = uuid.uuid4()
        detections = client.GetDetections(img_id, frame)

        # Get best/closer detections when multiple instances
        isHand = False
        centre = [x/2 for x in frame_size]
        print(centre)

        lastArea = 0
        target = []
        for detection in detections:
            if detection[class_id] == targetClass:
                area = get_detection_area(detection)
                if area > lastArea:
                    lastArea = area
                    target = detection
            elif detection[class_id] == CLASSES["hand"]:
                centre = get_detection_centre(detection)
                isHand = True

        # If object not in sight, play error
        if not target:
            play_error()
            time.sleep(0.1)
            continue

        # Get target vector
        vector_hor = target(0) - centre(0)
        vector_ver = target(1) - centre(1)

        horizontal_tolerance = frame_size(0)/3
        vertical_tolerance = frame_size(1)/3

        # Queue up directions
        if vector_hor > horizontal_tolerance:
            spkr.Queue("Rigth")
        elif vector_hor < -horizontal_tolerance:
            spkr.Queue("Left")

        if vector_ver > vertical_tolerance:
            spkr.Queue("Down")
        elif vector_ver < -vertical_tolerance:
            spkr.Queue("Up")

        # Give directions
        spkr.Flush()
        time.sleep(0.1)