#!/usr/bin/python3
from socketUtils import *
from imutils.video import VideoStream
import datetime
import argparse
import re
import imutils
import time
import cv2
import socket
import pickle
import uuid

class InferenceClient:
    def __init__(self, host, port):
        self.Host = host
        self.Port = port
        self.Server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Server.connect((host, port))

    def GetDetections(self, img_id, frame):
        data = EvaluationData(img_id, frame)

        pkg = pickle.dumps(data)
        #self.Server.send(pkg)
        send_msg(self.Server, pkg)

        #data = self.Server.recv(2 ^ 13)
        data = recv_msg(self.Server)
        resp = pickle.loads(data)
        return resp

def ip_with_port(arg):
    if re.match(r'\d{1,3}[.]\d{1,3}[.]\d{1,3}[.]\d{1,3}[:]\d+', arg):
        return arg
    raise argparse.ArgumentTypeError('Server address must be follow the format \'0.0.0.0:<port>\'')

# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--server", required=True, type=ip_with_port, help="IP and port of the server being used for detection. Required format - 0.0.0.0:<port>")
parser.add_argument("-P", "--picamera", action="store_true", help="Whether or not the Raspberry Pi camera should be used")

args = parser.parse_args()

host, port = args.server.split(':')
port = int(port)
client = InferenceClient(host, port)
#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect((host, port))

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(usePiCamera=args.picamera).start()
time.sleep(2.0)

#while True:
frame = vs.read()
img_id = uuid.uuid4()
resp = client.GetDetections(img_id, frame)

    #TODO: Process detection


