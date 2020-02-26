import multiprocessing as mp
import dlib
import cv2
import imutils
import time
import numpy as np
import os
import sys
import traceback 
import random

from String_to_Path import ThingToWrite
import URBasic
from imutils.video import VideoStream
import math3d as m3d
import math
from keras.preprocessing.image import img_to_array
#from keras.models import load_model

RASPBERRY_BOOL = False
# If this is run on a linux system, it is assumed it runs on a raspberry pi and a picamera will be used.
# If you are using a linux system, with a webcam instead of a raspberry pi delete the following if-statement
if sys.platform == "linux":
    import picamera
    from picamera.array import PiRGBArray
    import RPi.GPIO as GPIO
    RASPBERRY_BOOL = True

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

#emotion_classifier = load_model(emotion_model_path, compile=False)
face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

# Path to the face-detection model:
pretrained_model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
#pretrained_model = cv2.dnn.readNetFromCaffe("models/RFB-320.prototxt", "models/RFB-320.caffemodel")

video_resolution = (700, 400)  # resolution the video capture will be resized to, smaller sizes can speed up detection
video_midpoint = (int(video_resolution[0]/2),
                  int(video_resolution[1]/2))
video_asp_ratio  = video_resolution[0] / video_resolution[1]  # Aspect ration of each frame
video_viewangle_hor = math.radians(25)  # Camera FOV (field of fiew) angle in radians in horizontal direction
#video_viewangle_vert = video_viewangle_hor / video_asp_ratio  #  Camera FOV (field of fiew) angle in radians in vertical direction
m_per_pixel = 00.00009  # Variable which scales the robot movement from pixels to meters.


vs = VideoStream(src= 0 ,
                 usePiCamera= RASPBERRY_BOOL,
                 resolution=video_resolution,
                 framerate = 13,
                 meter_mode = "backlit",
                 exposure_mode ="auto",
                 shutter_speed = 8900,
                 exposure_compensation = 2,
                 rotation = 0).start()
print("wait ")
time.sleep(5)
i=0
while i< 3000:
    timer = time.time()
    frame = vs.read()
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    pretrained_model.setInput(blob)
    detections = pretrained_model.forward()
    print(detections.shape)
    print(time.time()-timer, "seconds")
    print("____"*20)
    i+=1