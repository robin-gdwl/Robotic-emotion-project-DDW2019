import multiprocessing as mp
import dlib
import cv2
import imutils
import time
import numpy as np
import os
import sys
from String_to_Path import ThingToWrite
import URBasic

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.preprocessing.image import img_to_array
from keras.models import load_model

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

emotion_classifier = load_model(emotion_model_path, compile=False)
face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]


class Face:
    def __init__(self, image):

        self.face_image = image
        self.emotions = {}
        self.landmarks = []
        self.face_target_size = 0.1
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
        print("initialised in:", time.time() - timer)

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
                for n in range(1, 4):
                    emotion = EMOTIONS[srtd_lst[-n]]
                    prob = preds[srtd_lst[-n]] * 100
                    text = "{}-{:.0f}%".format(emotion, prob)
                    # print(text)
                    emotion_results.append(text)
                print(emotion_results)

                return emotion_results

            else:
                i += 1
            cv2.imshow("Frame", frame)
            cv2.waitKey(1000)  # this defines how long each frame is shown

        person_emo = ["ERROR - 0 %", "_ _ _ _ _",
                      "ARE YOU ", "A ROBOT", "- ??? -"]
        return person_emo
        return emos

    def get_landmarks(self):
        lndmks = []

        faces = self.detector(self.gray)
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
        for n in range(0, 68):
            # apply face_scale and overall scale, move to upper left corner and offset by the origin
            x = (landmarks.part(n).x - face.left()) * face_scale
            y = (landmarks.part(n).y - face.top()) * face_scale

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

        self.landmarks = feature_lines

        return feature_lines


class Robot:
    def __init__(self, ip):
        self.ip = ip
        self.robot = None
        self.position = [0, 0]

        self.face_row_offset = [0, 0.04]
        self.text_hor_offset = 0.03
        self.z_hop = -0.03
        self.current_row = 0
        self.robotModel = URBasic.robotModel.RobotModel()
        

    def initialise_robot(self):
        self.robot = URBasic.urScriptExt.UrScriptExt(host=self.ip, robotModel=self.robotModel)
        pass

    def start_rtde(self):
        pass

    def wander(self):
        # use self.position to go to a close position and search for faces
        pass

    def follow_face(self, close = True):
        # either breaks or returns a face object if run for enough time
        pass

    def move_to_write(self):
        pass

    def move_home(self):
        pass

    def create_coordinates(self, face):
        pass

    def write_emotions(self, position, Face):
        if len(Face.emotions) == 0:
            print("no emotions to write")
            return False
        else:
            for emotion in Face.emotions:
                emotion_coords = ThingToWrite(emotion).string_to_coordinates()  # add origin here
                # set origin 
            origin = self.calculate_origin(text = True)
            self._draw_curves(emotion_coords, origin)
            return True

    def draw_face(self, position, Face):
        if len(Face.landmarks) == 0:
            print("no landmarks to draw")
            return False
        else:
            origin = self.calculate_origin()
            self._draw_curves(Face.landmarks, origin)
            return True

    def _draw_curves(self, polylines, origin_point):
        polylines_zvalue = self._add_zvalue(polylines)
        polylines_with_zhop = self._add_zhop(polylines_zvalue)
        polylines_rotvec = self._add_rotvec(polylines_with_zhop)
        for line in polylines_rotvec:
            self.robot.movel_waypoints(line)
            
        
        pass
    
    def calculate_origin(self, text = False): 
        row = self.current_row
        x = self.face_row_offset[0]
        y = (row + 0)* self.face_row_offset[1]  # TODO: do I need a general offset here? 
        
        if text:
            x += self.text_hor_offset
            
        origin = [x,y]
        return origin
    
    def _add_zvalue(self, list):
        
        lines_with_z = [coord.append(0) for line in list for coord in line]
        print("lines with added z: ", lines_with_z)
                 
        return lines_with_z
        
    def _add_zhop(self, list):
        for line in list:
            for coord in line:
                if len(coord)!= 3:
                    print("coordinate is missing a third value")
            line.insert(0, line[0].copy())
            line.append(line[-1].copy())
            line[0][2] = self.z_hop
            line[-1][2] = self.z_hop
        return list
    
    def _add_rotvec(self, list):
        lines_with_rotvec = [coord.append(0,0,0) for line in list for coord in line]
        print("lines with added rotation vector: ", lines_with_rotvec)

        return lines_with_rotvec
    
    
    def _string_to_coords(self):
        pass
        
    
    
def check_exhibit_time():
    pass


robot_ip = "10.211.55.5"
robot = Robot(robot_ip)
robot.move_home()

robot.current_row = 0
while True:
    robot.wander()
    face = robot.follow_face(close=True)

    landmark_queue = mp.Queue()
    emotion_queue = mp.Queue()

    coordinates = mp.Process(target=robot.create_coordinates, args=(face))  # evaluates emotion, creates landmarks returns coordinate list of face drawing and emotion text
    movement = mp.Process(target=robot.move_to_write, args=(current_pos))

    coordinates.start()
    movement.start()

    coordinates.join()
    movement.join()

    landmarks = landmark_queue.get()
    emotions = emotion_queue.get()
    robot.draw_face(landmarks)
    robot.write_emotions(emotions)

    robot.check_paper(current_pos)

    current_pos += 1
    robot.move_home()