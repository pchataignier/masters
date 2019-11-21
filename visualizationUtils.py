import cv2
import numpy as np
from mrcnn.visualize import random_colors, apply_mask
import datetime

CLASS_LABELS = {1:'phone', 2:'mug', 3:'remote', 4:'bottle', 5:'hand'}

def apply_results(image, results):
	colors = random_colors(len(results))
	for result, color in zip(results, colors):
		bgr_color_f = np.array(color)[[2, 1, 0]]
		bgr_color = (bgr_color_f * 255).astype("uint8")
		bgr_color = [int(c) for c in bgr_color]

		# Draw ROI
		y1, x1, y2, x2 = result["roi"]
		image = cv2.rectangle(image, (x1, y1), (x2, y2), bgr_color, 2)

		# Write Class label and score
		text = "{}: {:.4f}".format(CLASS_LABELS[result["class_id"]], result["score"])
		cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4)
		cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)

		# Draw mask
		image = apply_mask(image, result["mask"], bgr_color_f)

	return image

class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		if self._end:
			return (self._end - self._start).total_seconds()
		else:
			return (datetime.datetime.now() - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()