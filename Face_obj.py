from keras.preprocessing.image import img_to_array
from keras.models import load_model
import dlib
import cv2
import imutils
import time
import math
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

emotion_classifier = load_model(emotion_model_path, compile=False)
face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

# Path to the face-detection model:
pretrained_model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
pretrained_model2 = cv2.dnn.readNetFromCaffe("models/RFB-320.prototxt", "models/RFB-320.caffemodel")


class Face:
    def __init__(self, image, roi_boxes):

        self.face_image = image
        self.annotated_image = image
        self.rois = roi_boxes
        self.emotions = {}
        self.landmarks = []
        self.face_target_size = 140
        self.scale = 1 / 3000
        print("creating gray")
        # print(self.face_image.shape)
        self.gray = cv2.cvtColor(self.face_image, cv2.COLOR_BGR2GRAY)

    def evaluate(self):
        self.evaluate_emotions()
        self.get_landmarks()

    def evaluate_emotions(self):
        timer = time.time()
        print("emotion detection:")
        frame = self.face_image
        frame = imutils.resize(frame, width=300)
        faces = face_detection.detectMultiScale(self.gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                flags=cv2.CASCADE_SCALE_IMAGE)
        # print("initialised in:  ", time.time() - timer)

        time.sleep(0.1)
        i = 0
        while i <= 10:
            if len(faces) > 0:
                faces = sorted(faces, reverse=True,
                               key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                (fX, fY, fW, fH) = faces
                # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
                roi = self.gray[fY:fY + fH, fX:fX + fW]
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
                for n in range(1, 3):
                    emotion = EMOTIONS[srtd_lst[-n]]
                    prob = preds[srtd_lst[-n]] * 100
                    text = "{}-{:.0f} %".format(emotion, prob)
                    # print(text)
                    emotion_results.append(text)
                print("emotion_results. ", emotion_results)

                self.emotions = emotion_results
                print("emotions detected in:  ", time.time() - timer)
                return emotion_results  # TODO: put this to the end 

            else:
                i += 1
            

        person_emo = ["ERROR - 0 %", "_ _ _ _ _",
                      "ARE YOU ", "A ROBOT", "- ??? -"]
        print("Error: no emotions detected in:  ", time.time() - timer)
        self.emotions = person_emo
        return person_emo

    def get_landmarks(self):
        lndmks = []

        dlib_faces = detector(self.gray)
        faces = dlib.rectangles()
        for face_box in self.rois:
            f = dlib.rectangle(face_box[0], face_box[1], face_box[2], face_box[3])
            faces.append(f)

        print("__" * 20)
        print(faces)
        print(dlib_faces)
        print("__" * 20)

        if len(faces) > 0:
            face = faces[0]
            lndmks = predictor(self.gray, face)

            # scale the face coordinates and move them to the upper left corner:
            face_width = face.right() - face.left()
            face_height = face.bottom() - face.top()
            face_average = (face_width + face_height) / 2
            print("face width and height: ", face_width, " , ", face_height)
            # print("face average size: ", face_average)
            face_scale = self.face_target_size / face_average

            lndmk_points = []
            frame = self.annotated_image
            for n in range(0, 68):
                # print(lndmks.part(n))
                x = int((lndmks.part(n).x - face.left()) * face_scale)
                y = int((lndmks.part(n).y - face.top()) * face_scale)
                cv2.circle(frame, (x, y), 3, (100, 100, 255), -1)

                # apply face_scale and overall scale, move to upper left corner and offset by the origin
                x = (lndmks.part(n).x - face.left()) * face_scale * self.scale
                y = (lndmks.part(n).y - face.top()) * face_scale * self.scale

                lndmk_points.append([x, y])

            # individual features, each of these will be one continuous line to be drawn
            jawline = lndmk_points[:17]
            left_brow = lndmk_points[17:22]
            right_brow = lndmk_points[22:27]
            nose_ridge = lndmk_points[27:31]
            nose_tip = lndmk_points[31:36]

            left_eye = lndmk_points[36:42]
            left_eye.append(left_eye[0].copy())  # add the first point to get a closed curve

            right_eye = lndmk_points[42:48]
            right_eye.append(right_eye[0].copy())  # add the first point to get a closed curve

            lips_outer = lndmk_points[48:60]
            lips_outer.append(lips_outer[0].copy())  # add the first point to get a closed curve

            lips_inner = lndmk_points[60:68]
            lips_inner.append(lips_inner[0].copy())  # add the first point to get a closed curve

            feature_lines = [jawline,
                             left_brow,
                             right_brow,
                             nose_ridge,
                             nose_tip,
                             left_eye,
                             right_eye,
                             lips_inner,
                             lips_outer]
            # feature_lines is now a list of all lines to be drawn each list consisting of a list of coordinates
            self.annotated_image = frame
            #cv2.imshow("current", frame)
            #cv2.waitKey(10)  # this defines how long each frame is shown
            self.landmarks = feature_lines
            # print(self.landmarks)
            return feature_lines

        else:
            print("landmarks not detected")
            return [[[0, 0]]]

