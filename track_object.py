import os
import re
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from speaker import Speaker
from imutils.video import VideoStream
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import sounddevice as sd
from scipy.io.wavfile import read

GROUPINGS = {"Mobile phone":"phone", "Corded phone":"phone", "Telephone":"phone", "Remote control":"remote"}
error_sampling, error_wav=read('error.wav')


## Auxiliary Functions
def remap_classification_name(classification):
    return GROUPINGS.get(classification, classification)

def play_error():
    sd.play(error_wav,error_sampling)
    #os.system('aplay -q error.wav')

def get_box_centre(box):
    ymin, xmin, ymax, xmax = box
    return ( ((xmax - xmin) / 2) + xmin, ((ymax - ymin) / 2) + ymin )

def get_box_area(box):
    ymin, xmin, ymax, xmax = box
    return (xmax - xmin) * (ymax - ymin)

## Construct the argument parse and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label_map", required=True, help="Path to the '.pbtxt' Label Map file")
parser.add_argument("-m", "--model", required=True, help="Path to the '.pb' inference graph file")
parser.add_argument("-t", "--target", required=True, help="Object to track")
parser.add_argument("-v", "--visualize", action="store_true", help="Whether or not to display the captured video and inference")
args = parser.parse_args()

LABEL_MAP = args.label_map
PATH_TO_CKPT = args.model

# faster_rcnn = "../models/frozen-faster-rcnn-oid-1081/frozen_inference_graph.pb"
# mobilenet = "../models/frozen-mobilenet-oid-7869/frozen_inference_graph.pb"
# mobile_coco = "../models/frozen-mobilenet-fpn-coco-5981/frozen_inference_graph.pb"
# resnet = "../models/frozen-resnet101-oid-8466/frozen_inference_graph.pb"
#
# #DATASET_DIR = "./OpenImagesDataset"
# MODELS_DIR = "../models/"
# LABEL_MAP = "./OpenImagesDataset/labelMap.pbtxt" #oid_labelMap #labelMap
# PATH_TO_CKPT = f"../models/{mobile_coco}/frozen_inference_graph.pb"

spkr = Speaker()
category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP, use_display_name=True)


## Load detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

camera = VideoStream().start()
time.sleep(2.0)

## Main inference loop
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        try:
            while True:
                frame = camera.read()
                frame_size = frame.shape

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(frame, axis=0)

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


                # Get best/closer detections when multiple instances
                isHand = False
                centre = [0.5, 0.5] #[x / 2 for x in frame_size]
                #print(centre)

                lastArea = 0
                target = []
                for i in range(num_detections):
                    pred_class = category_index[classes[i]]["name"]
                    pred_class = GROUPINGS.get(pred_class, pred_class)
                    box = boxes[i]
                    if pred_class == args.target:
                        area = get_box_area(box)
                        if area > lastArea:
                            lastArea = area
                            target = get_box_centre(box)
                    elif pred_class == "Human hand":
                        centre = get_box_centre(box)
                        isHand = True

                # If object not in sight, play error
                if not target:
                    play_error()
                    time.sleep(0.1)
                else:
                    # Get target vector
                    vector_hor = target(0) - centre(0)
                    vector_ver = target(1) - centre(1)

                    horizontal_tolerance = 0.33 #frame_size(0) / 3
                    vertical_tolerance = 0.33 #frame_size(1) / 3

                    # Queue up directions
                    if vector_hor > 1-horizontal_tolerance:
                        spkr.Queue("Rigth")
                    elif vector_hor < horizontal_tolerance:
                        spkr.Queue("Left")

                    if vector_ver > 1-vertical_tolerance:
                        spkr.Queue("Down")
                    elif vector_ver < vertical_tolerance:
                        spkr.Queue("Up")

                    # Give directions
                    spkr.Flush()

                if args.visualize:
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        frame,
                        boxes,
                        classes,
                        scores,
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)

                    cv2.imshow('object detection', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        break

                #time.sleep(0.1)
        finally:
            camera.stop()