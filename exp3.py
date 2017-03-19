import numpy as np
import cv2
import dlib
import numpy
import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
# detector2= face_recognition_model_v1() 
predictor = dlib.shape_predictor(PREDICTOR_PATH)

cap = cv2.VideoCapture('video-1488820746.mp4')


while(cap.isOpened()):
    ret, frame = cap.read()
    rects = detector(frame, 0)
    if len(rects) == 1: 
	    for i in  predictor(frame, rects[0]).parts() :
			cv2.circle(frame,tuple([i.x,i.y]), 3, (0,0,255), -1)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()












