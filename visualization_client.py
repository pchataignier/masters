#!/usr/bin/python3
from socketUtils import *
from imutils.video import VideoStream
import argparse
import re
import time
import socket
import pickle
import uuid
import os
from visualizationUtils import apply_results
import cv2

CLASSES = {'telephone':1, 'mug':2, 'remote control':3, 'remote':3, 'bottle':4, 'hand':5}
CLASS_LABELS = {1:'phone', 2:'mug', 3:'remote', 4:'bottle', 5:'hand'}

class_id = "class_id"
mask = "mask"
roi = "roi" #ROI - y1, x1, y2, x2
score = "score"

class InferenceClient:
    def __init__(self, host, port):
        self.Host = host
        self.Port = port
        self.Server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.Server.connect((host, port))

    def GetDetections(self, img_id, frame, zipResults=True):
        data = EvaluationData(img_id, frame, zipResults)

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

def get_detection_centre(detection):
    return ( ((detection[roi][3] - detection[roi][1]) / 2) + detection[roi][1], ((detection[roi][2] - detection[roi][0]) / 2) + detection[roi][0] )

def get_detection_area(detection):
    return (detection[roi][3] - detection[roi][1]) * (detection[roi][2] - detection[roi][0])


# construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--server", required=True, type=ip_with_port, help="IP and port of the server being used for detection. Required format - 0.0.0.0:<port>")
parser.add_argument("-i", "--image", required=False, help="Path to image file")

args = parser.parse_args()

host, port = args.server.split(':')
host = socket.gethostbyname(socket.gethostname())
port = int(port)
client = InferenceClient(host, port)

if os.path.exists(args.image):
    image = cv2.imread(args.image)
    image = cv2.resize(image, (640, 480))

    try:
        img_id = uuid.uuid4()
        InferenceStart = time.perf_counter()
        detections = client.GetDetections(img_id, image, zipResults=True)
        InferenceStop = time.perf_counter()

        DrawStart = time.perf_counter()
        results = detections.data
        clone_img = image.copy()
        # Draw bbox and masks
        clone_img = apply_results(clone_img, results)
        DrawStop = time.perf_counter()

        print("Inference time: {:.6f}".format(InferenceStop-InferenceStart))
        print("Draw time: {:.6f}".format(DrawStop - DrawStart))
        cv2.imshow("Inference", clone_img)
        cv2.waitKey(0)
    finally:
        client.Close()


if args.image is None:
    # initialize the video stream and allow the cammera sensor to warmup
    vs = VideoStream().start()
    time.sleep(2.0)

    while True:  # TODO: stop when found
        # Get frame and run it through inferece
        frame = vs.read()
        frame_size = frame.shape
        img_id = uuid.uuid4()
        detections = client.GetDetections(img_id, frame)