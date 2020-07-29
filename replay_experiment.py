import os
import cv2
import time
import json
import argparse
import numpy as np

from glob import glob
from chirp import Chirp
from beeper import Beeper
from speaker import Speaker
from datetime import datetime
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import sounddevice as sd
from scipy.io.wavfile import read as read_wav

error_sampling, error_wav=read_wav("error.wav")
def play_error():
    sd.play(error_wav,error_sampling)

ready_sampling, ready_wav=read_wav("ready.wav")
def play_ready():
    sd.play(ready_wav,ready_sampling)

def get_output(mode, guide_tolerance, guide_delay):
    guide = None
    # guide_tolerance = 0.15
    # guide_delay = 0
    if mode == "speaker":
        guide = Speaker(tolerance=guide_tolerance, delay=guide_delay)
    elif mode == "beeper":
        guide = Beeper(tolerance=guide_tolerance, delay=guide_delay)
    elif mode == "chirp":
        guide = Chirp(tolerance=guide_tolerance, delay=guide_delay)

    return guide

def rel_to_abs(coord, size):
    return tuple([abs for abs in map(lambda x: int(x[0]*x[1]), zip(coord, size))])

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True, type=str, help="Path to experiment Log directory or file.")
parser.add_argument("--manual", action="store_true", help="Whether or not to record experiment")
args = parser.parse_args()

LOG_FILE = None
LOG_DIR = None

if os.path.isdir(args.log):
    LOG_FILE = glob(os.path.join(args.log, "*.log"))[0]
    LOG_DIR = args.log
elif os.path.isfile(args.log) and os.path.exists(args.log):
    LOG_FILE = args.log
    LOG_DIR = os.path.dirname(os.path.abspath(args.log))


logs = None
with open(LOG_FILE) as f:
    logs = f.readlines()


config = json.loads(logs.pop(0))
TARGET = config["target_class"]
OUT_MODE = config["output_mode"]
TOLERANCE = config["guide_tolerance"]

category_index = label_map_util.create_category_index_from_labelmap(config["label_map"], use_display_name=True)
guide = get_output(OUT_MODE, TOLERANCE, config["guide_delay"])

records = json.loads('['+','.join(logs)+']')
try:
    last_frame_time = None
    last_frame_processing = 0
    for record in records:
        frame_id = record["frame_id"]
        frame_time = datetime.strptime(frame_id, '%Y%m%d_%H%M%S_%f.jpg')

        if last_frame_time is not None:
            fps_time = frame_time - last_frame_time
            fps_time = fps_time.total_seconds()
        else:
            fps_time = 0
        to_wait = (fps_time - last_frame_processing) * 1000
        to_wait = int(to_wait)
        key = cv2.waitKey(to_wait if to_wait > 0 else 1)

        processing_start = time.perf_counter()
        if key == ord('q'):
            break
        frame = cv2.imread(os.path.join(LOG_DIR, frame_id))

        boxes = np.array(record["boxes"])
        classes = np.array(record["classes"])
        scores = np.array(record["scores"])
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            boxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4, min_score_thresh=0.4)

        h,w,_ = frame.shape
        frame_size = (w,h)
        target = tuple(record["target"])
        target_abs = rel_to_abs(target, frame_size)
        centre = tuple(record["centre"])
        centre_abs = rel_to_abs(centre, frame_size)
        found_once_before = record["found_once_before"]
        # print(f"Shape: {frame_size}\nTarget: {target}\n Target Abs: {target_abs}\nCentre: {centre}\n Centre Abs: {centre_abs}")

        # Specify Target
        label_text = f"LF: [{TARGET}]"
        text_size,_ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
        frame = cv2.rectangle(frame, (3, 12 - text_size[1]), (7 + text_size[0], 28), (0,0,0), -1)
        frame = cv2.putText(frame, label_text, (5,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

        # Draw Box Centres (target and centre)
        _circle_radius = 5
        _circle_thickness = 2
        if target:
            frame = cv2.circle(frame, target_abs, _circle_radius, (0,0,255), _circle_thickness)
            frame = cv2.circle(frame, centre_abs, _circle_radius, (0, 0, 255), _circle_thickness)
            frame = cv2.arrowedLine(frame, centre_abs, target_abs, (255, 0, 0), _circle_thickness)

        # Draw quadrants
        y_upper = rel_to_abs([centre[1] - TOLERANCE], [h])[0]
        y_lower = rel_to_abs([centre[1] + TOLERANCE], [h])[0]
        frame = cv2.line(frame, (0, y_upper), (w, y_upper), (255, 255, 255), 1, 4)
        frame = cv2.line(frame, (0, y_lower), (w, y_lower), (255, 255, 255), 1, 4)

        if OUT_MODE == "speaker":
            x_left = rel_to_abs([centre[0] - TOLERANCE], [w])[0]
            x_right = rel_to_abs([centre[0] + TOLERANCE], [w])[0]
            frame = cv2.line(frame, (x_left, 0), (x_left, h), (255, 255, 255), 1, 4)
            frame = cv2.line(frame, (x_right, 0), (x_right, h), (255, 255, 255), 1, 4)

        cv2.imshow("Experiment Replay", frame)
        cv2.waitKey(1)

        # Play outputs
        if not target:
            if found_once_before:
                play_ready()
            else:
                play_error()
            time.sleep(0.2)
        else:
            guide.give_directions(target, centre)

        processing_stop = time.perf_counter()
        last_frame_processing = processing_stop - processing_start
        last_frame_time = frame_time

    cv2.waitKey(0)
finally:
    cv2.destroyAllWindows()