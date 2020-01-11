import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import dlib
import cv2
import time
import imutils
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from imutils.video import VideoStream

RASPBERRY_BOOL = False

if sys.platform == "linux":
    import picamera
    from picamera.array import PiRGBArray
    RASPBERRY_BOOL = True


# TODO: use imutils videostream instead of a single image capture!!!
    # TODO: camera args as described here: https://github.com/jrosebr1/imutils/blob/master/imutils/video/videostream.py#L6
# camera stream will work in the bg

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

class FaceOperation:

    def __init__(self):

        self.landmarks = []
        self.emotion = []
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        #self.cap = cv2.VideoCapture(0)
        #self.screen_width = self.cap.get(3)   # x- extent of the captured frame
        #self.screen_height = self.cap.get(4)  # y- extent
        self.face_loc = []
        #self.camera = picamera.PiCamera()
        #self.camera.resolution = (800, 800)
        #self.rawCapture = PiRGBArray(self.camera)
        self.frame = None
        self.vs = VideoStream(usePiCamera= RASPBERRY_BOOL,
                              resolution=(1080, 720),
                              framerate = 16,
                              meter_mode = "backlit",
                              exposure_mode ="backlight",
                              shutter_speed = 16000).start()
        time.sleep(0.2)

    def getframe(self):
        # set camera options to guarantee good picture:
        #self.camera.exposure_mode = "backlight"
        #self.camera.meter_mode = "backlit"
        #self.camera.shutter_speed = 0

        timer = time.time()
        #self.camera.capture(self.rawCapture, format="bgr")  # capture the image
        frame = self.vs.read()
        self.frame = frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

        #image = self.rawCapture.array
        # print("image taken in", time.time() - timer)
        #self.rawCapture.truncate(0)
        # print("image processed in", timer - time.time())
        #print("exposure time: ", self.camera.exposure_speed)
        return frame

    def findface(self):
        # returns boolean weather or not a face is found
        frame = self.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Frame", frame)
        cv2.waitKey(10)  # this defines how long each frame is shown

        faces = self.detector(gray)

        if len(faces) >= 1:
            return True
        else:
            return False

            # track_face(face_x, face_y, screen_width, screen_height) # ignore this for now

    def facelocation(self):
        # returns location (x,y) of the face if one is detected


        frame = self.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)
        cv2.imshow("Frame", frame)
        cv2.waitKey(10)  # this defines how long each frame is shown

        if len(faces) >= 1:
            face_to_eval = faces[0]
            # print(face_to_eval)
            face_pos = face_to_eval.center()

            face_x = face_pos.x
            face_y = face_pos.y
            face_screen_xy = [face_pos.x, face_pos.y]
            self.face_loc = face_screen_xy

            return True
        else:
            return False

    def landmark_detection(self, origin = [0,0], scale = 1 / 3000):
        # detects the landmarks of the face and returns them as a list of list of coordinates
        frame = self.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = self.detector(gray)
        # TODO: this shouldn't be called in each method. make it a parmeter, check if it is there, redo if necessary
        if len(faces) >= 1:
            face = faces[0]
            landmarks = self.predictor(gray, face)
            landmark_points = []
            face_scale = 1  # this will be used to scale each face to a similar size
            face_target_size = 140

            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # scale the face coordinates and move them to the upper left corner:
            face_width = face.right() - face.left()
            face_height = face.bottom() - face.top()
            face_average = (face_width + face_height) / 2
            print("face width and height: ", face_width, " , ", face_height)
            #print("face average size: ", face_average)
            face_scale = face_target_size / face_average


            for n in range(0, 68):
                # print(landmarks.part(n))
                x = int((landmarks.part(n).x - face.left()) * face_scale)
                y = int((landmarks.part(n).y - face.top()) * face_scale)
                cv2.circle(frame, (x, y), 3, (100, 100, 255), -1)

                # apply face_scale and overall scale, move to upper left corner and offset by the origin
                x = (landmarks.part(n).x - face.left()) * face_scale * scale + origin[0]
                y = (landmarks.part(n).y - face.top()) * face_scale * scale + origin[1]

                landmark_points.append([x, y])



            # individual features, each of these will be one continuous line to be drawn
            jawline = landmark_points[:17]
            left_brow = landmark_points[17:22]
            right_brow = landmark_points[22:27]
            nose_ridge = landmark_points[27:31]
            nose_tip = landmark_points[31:36]

            left_eye = landmark_points[36:42]
            left_eye.append(left_eye[0].copy()) # add the first point to get a closed curve

            right_eye = landmark_points[42:48]
            right_eye.append(right_eye[0].copy())

            lips_outer = landmark_points[48:60]
            lips_outer.append(lips_outer[0].copy())

            lips_inner = landmark_points[60:68]
            lips_inner.append(lips_inner[0].copy())

            feature_lines = [jawline, left_brow, right_brow, nose_ridge, nose_tip, left_eye, right_eye, lips_inner, lips_outer]
            # feature_lines is now a list of all lines to be drawn
            cv2.imshow("Frame", frame)
            cv2.waitKey(10)  # this defines how long each frame is shown

            # print("fl:    ", feature_lines)
            # for line in feature_lines:
            #    print("line no add: ", line)
            feature_lines = self.apply_zhop(feature_lines)
            # print("feature lines", feature_lines)
            single_coords = []
            single_coords = [item for sublist in feature_lines for item in sublist]
            # print("sc: ", single_coords)
            self.landmarks = single_coords

            return feature_lines

    def apply_zhop(self, list_of_lines, z_hop = -0.05):

        for line in list_of_lines:
            line.insert(0, line[0].copy())
            line.append(line[-1].copy())
            # print(line)
            for xy in line:
                # print(xy)
                xy.append(0)
                # xy = [a + [0] for a in xy]  # no idea why this doesnt work...
            line[0][2] = z_hop
            line[-1][2] = z_hop


        # print(list_of_lines)



            #line.insert(0, line[0])
            #line.append(line[-1])

        time.sleep(0.1)
        return list_of_lines

    def detect_emotion(self):

        timer = time.time()
        print("emotion detection:")
        frame = self.frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = imutils.resize(frame, width=300)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        print("initialised in:", time.time()-timer)

        time.sleep(0.1)
        i = 0
        while i <= 10:
            if len(faces) > 0:
                faces = sorted(faces, reverse=True,
                               key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
                roi = gray[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (64, 64))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = emotion_classifier.predict(roi)[0]
                emotion_probability = np.max(preds)
                emotion_list = EMOTIONS
                label = emotion_list[preds.argmax()]

                srtd_lst = np.argsort(preds)

                emotion_results = []
                for n in range(1,4):
                    emotion = EMOTIONS[srtd_lst[-n]]
                    prob = preds[srtd_lst[-n]] * 100
                    text = "{}-{:.0f}%".format(emotion, prob)
                    #print(text)
                    emotion_results.append(text)
                print(emotion_results)

                return emotion_results

            else:
                i += 1
            cv2.imshow("Frame", frame)
            cv2.waitKey(1000)  # this defines how long each frame is shown

        person_emo = ["ERROR - 0 %", "_ _ _ _ _",
                      "Algorithmic", "Emotion", "git.io/JeBaC"]
        return person_emo


        # detects the emotion and returns a list of three strings: ["most common emotion: 20%", ....]



        return None
