from threading import Thread
import sys
import cv2
 
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
	from queue import Queue
 
# otherwise, import the Queue class for Python 2.7
else:
	from Queue import Queue


class FileVideoStream:
	def __init__(self, path, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
 
		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)	

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
	# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				return
 
			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()
 
				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stop()
					return
 
				# add the frame to the queue
				self.Q.put(frame)

	def read(self):
		# return next frame in the queue
		return self.Q.get()
		
	def more(self):
		# return True if there are still frames in the queue
		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2



import time
import datetime as dt

import cv2
import dlib
import numpy

import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (list(range(0, 2)) + list(range(15, 27))  + list(range(38, 48)) + [30, 32 ] )
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
]

COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 0)
    
    if len(rects) != 1 : 
        return None   
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def get_image_mask(im):
	# mask = cv2.Mat(im.size(), cv2.CV_8UC3)	# TODO: Can change to 1 Channel
	mask = numpy.zeros((im.shape[0], im.shape[1], 3), dtype=numpy.uint8)
	for i in range(im.shape[0]):
		for j in range(im.shape[1]):
			mask_value = im[i, j, 3]/255
			mask[i, j] = [mask_value, mask_value, mask_value]
	return mask
    
def transformation_from_points(points1, points2):
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

# cap = cv2.VideoCapture(0)

# construct the argument parse and parse the arguments
 
# start the file video stream thread and allow the buffer to
# start to fill
fps = FPS().start()

frame = None 
temp_im = cv2.imread("Sunglasses-PNG-File.png", cv2.IMREAD_UNCHANGED)
temp_im2 = cv2.resize(temp_im, (720, 360))
mask = get_image_mask(temp_im2)	# TODO: Check the speed

im = cv2.imread("Sunglasses-PNG-File.png", cv2.IMREAD_COLOR)
im2 = cv2.resize(im, (720, 360))

landmarks2 = numpy.load("sunglasses.npy")
landmarks2 = numpy.asmatrix(landmarks2)



print("[INFO] starting video file thread...")
fvs = FileVideoStream(0).start()
time.sleep(1.0)
 
# start the FPS timer
prev_time = time.time()


while fvs.more():
	frame = fvs.read()
	im1 = frame 
	output_im = im1	# TODO: Redundant remove it
	landmarks1 = get_landmarks(im1)
	
	if landmarks1 != None :   
	
		M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
		warped_mask = warp_im(mask, M, im1.shape)	
		warped_im2 = warp_im(im2, M, im1.shape)
		output_im = (im1 * (1.0 - warped_mask) + warped_im2 * warped_mask)/255
	cv2.imshow('frame2',output_im)
	curr_time = time.time()
	time_diff = curr_time - prev_time
	print (1 / time_diff), format(fvs.stream.get(cv2.CAP_PROP_FPS))
	prev_time = curr_time

	# frame = imutils.resize(frame, width=450)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = np.dstack([frame, frame, frame])
 
	# display the size of the queue on the frame
	cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
 
	# show the frame and update the FPS counter
	# cv2.imshow("Frame", frame)
	cv2.waitKey(1)
	fps.update()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
fvs.stop()										