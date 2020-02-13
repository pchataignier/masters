#import os
import time

import cv2
#import pandas as pd
import tensorflow as tf
import numpy as np
from imutils.video import VideoStream

#from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#pd.set_option("display.max_colwidth", 10000)

# faster_rcnn = "faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12"
# mobilenet = "ssd_mobilenet_v2_oid_v4_2018_12_12"
# mobile_coco = "ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
# resnet = "ssd_resnet101_v1_fpn_shared_box_predictor_oid_512x512_sync_2019_01_20"

faster_rcnn = "frozen-faster-rcnn-oid-1081"
mobilenet = "frozen-mobilenet-oid-7869"
mobile_coco = "frozen-mobilenet-fpn-coco-5981"
resnet = "frozen-resnet101-oid-8466"

#DATASET_DIR = "./OpenImagesDataset"
MODELS_DIR = "../models/"
LABEL_MAP = "./OpenImagesDataset/labelMap.pbtxt" #oid_labelMap #labelMap
PATH_TO_CKPT = MODELS_DIR + f"{mobilenet}/frozen_inference_graph.pb"

category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP, use_display_name=True)

cap = VideoStream().start()
time.sleep(2.0)

## Load detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

## Main inference loop
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        try:
            while True:
                image_np = cap.read()

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

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

                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                  image_np,
                  boxes,
                  classes,
                  scores,
                  category_index,
                  use_normalized_coordinates=True,
                  line_thickness=8)

                cv2.imshow('object detection', image_np)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break
        finally:
            cap.stop()