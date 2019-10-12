import dlib
import cv2
import time

class FaceOperation:

    def __init__(self):

        self.landmarks = []
        self.emotion = []
        self.detector = dlib.get_frontal_face_detector()
        self.cap = cv2.VideoCapture(0)
        self.screen_width = self.cap.get(3)   # x- extent of the captured frame
        self.screen_height = self.cap.get(4)  # y- extent


    def findface(self):
        # returns boolean weather or not a face is found
        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)
        if len(faces) >= 1:
            return True
        else:
            return False

            # track_face(face_x, face_y, screen_width, screen_height) # ignore this for now

    def facelocation(self):
        # returns location (x,y) of the face if one is detected
        # TODO: combine this with findface() to not do the same thing twice

        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)
        if len(faces) >= 1:
            face_to_eval = faces[0]
            print(face_to_eval)
            face_pos = face_to_eval.center()

            face_x = face_pos.x
            face_y = face_pos.y
            face_screen_xy = [face_pos.x, face_pos.y]

            return face_screen_xy

    def landmark_detection(self):
        # detects the landmarks of the face and returns them as a list of list of coordinates
        return None

    def detect_emotion(self):
        # detects the emotion and returns a list of three strings: ["most common emotion: 20%", ....]
        return None