import dlib
import cv2
import time

class Faceoperation:

    def __init__(self):

        self.landmarks
        self.emotion


    def findface(self):
        # returns boolean weather or not a face is found

    def facelocation(self):
        # returns location (x,y) of the face if one is detected

    def landmark_detection(self):
        # detects the landmarks of the face and returns them as a list of list of coordinates

    def detect_emotion(self):
        # detects the emotion and returns a list of three strings: ["most common emotion: 20%", ....]