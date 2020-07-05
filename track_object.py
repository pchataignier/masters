import os
import re
import cv2
import time
import json
import argparse
import threading
import numpy as np
import tensorflow as tf
from chirp import Chirp
from beeper import Beeper
from speaker import Speaker
from datetime import datetime
from imutils.video import VideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import sounddevice as sd
from scipy.io.wavfile import read as read_wav

GROUPINGS = {"Mobile phone":"phone", "Corded phone":"phone", "Telephone":"phone", "Remote control":"remote", "Coffee cup": "Mug"}


## Auxiliary Functions
def remap_classification_name(classification):
    return GROUPINGS.get(classification, classification)

error_sampling, error_wav=read_wav("error.wav")
def play_error():
    sd.play(error_wav,error_sampling)

ready_sampling, ready_wav=read_wav("ready.wav")
def play_ready():
    sd.play(ready_wav,ready_sampling)

def get_box_centre(box):
    y_min, x_min, y_max, x_max = box
    return ( ((x_max - x_min) / 2) + x_min, ((y_max - y_min) / 2) + y_min )

def get_box_area(box):
    y_min, x_min, y_max, x_max = box
    return (x_max - x_min) * (y_max - y_min)

def get_input():
    global flag
    _=input('Press a key \n')
    # thread doesn't continue until key is pressed
    flag=False

def log_to_file(to_log, log_file):
    to_log = json.dumps(to_log)
    with open(log_file, 'a') as f:
        f.write(f"{to_log}\n")

## Construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label_map", required=True, help="Path to the '.pbtxt' Label Map file")
parser.add_argument("-m", "--model", required=True, help="Path to the '.pb' inference graph file")
parser.add_argument("-t", "--target", required=True, help="Object to track")
parser.add_argument("-v", "--visualize", action="store_true", help="Whether or not to display the captured video and inference")
parser.add_argument("-c", "--camera", required=False, type=int, default=0, help="Camera number")
parser.add_argument("-o", "--output_mode", required=True, choices=["speaker", "beeper", "chirp"], help="Output mode: [speaker, beeper, chirp]")
parser.add_argument("--fps", type=int, default=2, help="FPS of output video")
parser.add_argument("--codec", type=str, default="XVID", help="Codec of output video. Default: XVID")
parser.add_argument("--logs", type=str, default="./Experiments", help="Path to Log directory. Default: ./Experiments")
parser.add_argument("--record", action="store_true", help="Whether or not to record experiment")
args = parser.parse_args()

LABEL_MAP = args.label_map
PATH_TO_CKPT = args.model
TARGET_CLASS = args.target
SHOULD_VISUALIZE = args.visualize
CAMERA_ID = args.camera
OUTPUT_MODE = args.output_mode
LOGS = args.logs

CODEC = args.codec
FPS = args.fps
RECORD = args.record

if not os.path.exists(LOGS):
    os.mkdir(LOGS)
EXPERIMENT_ID = datetime.now().strftime('experiment_%Y%m%d_%H%M%S')
EXPERIMENT_LOG = os.path.join(LOGS, EXPERIMENT_ID)
os.mkdir(EXPERIMENT_LOG)

LOG_FILE = os.path.join(EXPERIMENT_LOG, EXPERIMENT_ID + ".log")

VIDEO_FILE = None
if RECORD:
    filename = EXPERIMENT_ID + ".avi"
    VIDEO_FILE = os.path.join(EXPERIMENT_LOG, filename)

## Define output mode
guide = None
guide_tolerance = 0.15
guide_delay = 0
if OUTPUT_MODE == "speaker":
    guide = Speaker(tolerance=guide_tolerance, delay=guide_delay)
elif OUTPUT_MODE == "beeper":
    guide = Beeper(tolerance=guide_tolerance, delay=guide_delay)
elif OUTPUT_MODE == "chirp":
    guide = Chirp(tolerance=guide_tolerance, delay=guide_delay)

experiment_config = {"label_map":LABEL_MAP, "model_ckpt":PATH_TO_CKPT, "target_class":TARGET_CLASS,
                     "output_mode":OUTPUT_MODE, "guide_tolerance":guide_tolerance, "guide_delay":guide_delay}
log_to_file(experiment_config, LOG_FILE)

category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP, use_display_name=True)

## Load detection graph
start = time.perf_counter()
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

stop = time.perf_counter()
print(f"Loading graph: {stop-start}")

## Camera startup
camera = VideoStream(CAMERA_ID).start()
time.sleep(1.0)

## Recording prep
fourcc = cv2.VideoWriter_fourcc(*CODEC)
video_writer = None

## Main inference loop
start = time.perf_counter()
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        stop = time.perf_counter()
        print(f"Loading session: {stop - start}")
        play_ready()
        flag=1
        i = threading.Thread(target=get_input)
        i.start()
        try:
            start = time.perf_counter()
            found_once_before = False
            first_frame = True
            fps_start = time.perf_counter()
            first_frame_delay = 0
            while flag==1:
                frame_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f.jpg')
                results_dict = {"frame_id":frame_id}
                frame = camera.read()
                #frame_size = frame.shape

                # if RECORD and not video_writer:
                #     (frame_h, frame_w) = frame.shape[:2]
                #     video_writer = cv2.VideoWriter(VIDEO_FILE, fourcc, FPS, (frame_w, frame_h), True)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame_rgb, axis=0)

                # Get input and output tensors
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection
                (boxes, scores, classes, num_detections) = sess.run(
                  [boxes, scores, classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})

                boxes = np.squeeze(boxes) # ymin, xmin, ymax, xmax = boxes[i]
                classes = np.squeeze(classes).astype(np.int32) # category_index[classes[i]]["name"]
                scores = np.squeeze(scores)
                num_detections = int(num_detections)

                results_dict["boxes"] = boxes[:num_detections].tolist()
                results_dict["classes"] = classes[:num_detections].tolist()
                results_dict["scores"] = scores[:num_detections].tolist()
                results_dict["num_detections"] = num_detections


                # Get best/closer detections when multiple instances
                isHand = False
                centre = [0.5, 0.5] #[x / 2 for x in frame_size]
                #print(centre)

                lastArea = 0
                target = []
                target_indx = None
                centre_inx = None
                for i in range(num_detections):
                    pred_class = category_index[classes[i]]["name"]
                    pred_class = remap_classification_name(pred_class)
                    box = boxes[i]
                    if pred_class == TARGET_CLASS and scores[i]>=0.5:
                        area = get_box_area(box)
                        if area > lastArea:
                            lastArea = area
                            target = get_box_centre(box)
                            target_indx = i
                    elif pred_class == "Human hand" and scores[i]>=0.5:
                        centre = get_box_centre(box)
                        centre_inx = i
                        isHand = True

                results_dict["target"] = target
                results_dict["target_indx"] = target_indx
                results_dict["centre"] = centre
                results_dict["centre_inx"] = centre_inx
                results_dict["isHand"] = isHand

                # If object not in sight, play error
                if not target:
                    if found_once_before:
                        play_ready()
                    else:
                        play_error()
                    time.sleep(0.1)
                else:
                    found_once_before = True
                    guide.give_directions(target, centre)

                results_dict["found_once_before"] = found_once_before
                if RECORD:
                    frame_file = os.path.join(EXPERIMENT_LOG, frame_id)
                    cv2.imwrite(frame_file, frame)
                    # to_log = str(results_dict)
                    log_to_file(results_dict, LOG_FILE)

                # Write detections to frame
                if first_frame:
                    fps_stop = time.perf_counter()
                    first_frame_delay = fps_stop - fps_start
                    #print(f"Seconds on first frame: {first_frame_delay}")
                    first_frame = False
                elif video_writer:
                    annotated_frame = frame
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        annotated_frame,
                        boxes,
                        classes,
                        scores,
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4, min_score_thresh=0.1)  # , min_score_thresh=.1
                    video_writer.write(annotated_frame)

                if SHOULD_VISUALIZE:
                    # Visualization of the results of a detection.

                    annotated_frame = frame
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        annotated_frame,
                        boxes,
                        classes,
                        scores,
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4, min_score_thresh=0.5)  # , min_score_thresh=.1

                    cv2.imshow('object detection', annotated_frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        # cv2.destroyAllWindows()
                        break

                #time.sleep(0.1)
            stop = time.perf_counter()
            guidance_time = stop - start
            print(f"Guidance Loop: {guidance_time}\n"
                  f"First Frame Processing: {first_frame_delay}\n"
                  f"Discounted Guidance: {guidance_time - first_frame_delay}")
        finally:
            camera.stop()
            if SHOULD_VISUALIZE:
                cv2.destroyAllWindows()

            if video_writer:
                video_writer.release()