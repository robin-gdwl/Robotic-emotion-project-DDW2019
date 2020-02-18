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
from imutils.video import VideoStream
import math3d as m3d
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

from keras.preprocessing.image import img_to_array
from keras.models import load_model

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

emotion_classifier = load_model(emotion_model_path, compile=False)
face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised",
            "neutral"]

# Path to the face-detection model:
pretrained_model = cv2.dnn.readNetFromCaffe("MODELS/deploy.prototxt.txt", "MODELS/res10_300x300_ssd_iter_140000.caffemodel")

video_resolution = (700, 400)  # resolution the video capture will be resized to, smaller sizes can speed up detection
video_midpoint = (int(video_resolution[0]/2),
                  int(video_resolution[1]/2))
video_asp_ratio  = video_resolution[0] / video_resolution[1]  # Aspect ration of each frame
video_viewangle_hor = math.radians(25)  # Camera FOV (field of fiew) angle in radians in horizontal direction
#video_viewangle_vert = video_viewangle_hor / video_asp_ratio  #  Camera FOV (field of fiew) angle in radians in vertical direction
m_per_pixel = 00.00009  # Variable which scales the robot movement from pixels to meters.

RASPBERRY_BOOL = False
# If this is run on a linux system, a picamera will be used.
# If you are using a linux system, with a webcam instead of a raspberry pi delete the following if-statement
if sys.platform == "linux":
    import picamera
    from picamera.array import PiRGBArray
    RASPBERRY_BOOL = True

vs = VideoStream(src= 0 ,
                 usePiCamera= RASPBERRY_BOOL,
                 resolution=video_resolution,
                 framerate = 13,
                 meter_mode = "backlit",
                 exposure_mode ="auto",
                 shutter_speed = 8900,
                 exposure_compensation = 2,
                 rotation = 0).start()


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
        self.robotUR = None
        self.position = [0, 0]

        self.face_row_offset = [0, 0.04]
        self.text_hor_offset = 0.03
        self.z_hop = -0.03
        self.current_row = 0
        self.robotURModel = URBasic.robotModel.RobotModel()
        self.max_x = 0.2
        self.max_y = 0.2
        self.hor_rot_max = math.radians(50)
        self.vert_rot_max = math.radians(25)
        self.accel = 0.9
        self.vel = 0.5
        self.curr_origin = m3d.Transform()
        self.follow_time = 5 

    def initialise_robot(self):
        self.robotUR = URBasic.urScriptExt.UrScriptExt(host=self.ip, robotModel=self.robotURModel)
        pass

    def start_rtde(self):
        self.robotUR.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
        time.sleep(1)  # just a short wait to make sure everything is initialised
        print("rtde started")
        pass

    def wander(self):
        print("wander function")
        # use self.position to go to a close position and search for faces
        pass

    def follow_face(self, close = True):
        # either breaks or returns a face object if run for enough time
        try:
            print("starting loop")
            timer = time.time()
            while True:

                frame = vs.read()
                face_positions, new_frame = self.find_faces_dnn(frame)
                self.show_frame(new_frame)
                if len(face_positions) > 0:
                    if time.time() -timer < self.follow_time:
                        self.position = self.move_to_face(face_positions, self.position)
                    else:
                        print("time up returning frame")
                        return frame          
                else:
                    break
                #print("end of loop")
            print("exiting loopwithout face ")
            return False
        
        except KeyboardInterrupt:
            print("face tracking interrupted by user")
            print("closing robot connection")
            # Remember to always close the robot connection, otherwise it is not possible to reconnect
            self.robotUR.close()

        except:
            print("error during facetracking")
            self.robotUR.close()

    def move_to_write(self, row):
        pass

    def move_home(self):
        self.robotUR.movej(q=(math.radians(-218),
                       math.radians(-63),
                       math.radians(-93),
                       math.radians(-20),
                       math.radians(88),
                       math.radians(0)), a=self.accel, v=self.vel)

        self.position = [0, 0]
        self.origin = self.set_lookorigin()

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
            self.robotUR.movel_waypoints(line)
            
        
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

    def find_faces_dnn(self,image):
        """
        Finds human faces in the frame captured by the camera and returns the positions
        uses the pretrained model located at pretrained_model

        Input:
            image: frame captured by the camera

        Return Values:
            face_centers: list of center positions of all detected faces
                list of lists with 2 values (x and y)
            frame: new frame resized with boxes and probabilities drawn around all faces

        """

        frame = image
        frame = imutils.resize(frame, width=video_resolution[0])

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and predictions
        pretrained_model.setInput(blob)

        # the following line handles the actual face detection
        # it is the most computationally intensive part of the entire program
        # TODO: find a quicker face detection model
        detections = pretrained_model.forward()
        face_centers = []
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.4:
                continue

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            face_center = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))
            position_from_center = (face_center[0] - video_midpoint[0], face_center[1] - video_midpoint[1])
            face_centers.append(position_from_center)

            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # cv2.putText(frame, str(position_from_center), face_center, 0, 1, (0, 200, 0))
            cv2.line(frame, video_midpoint, face_center, (0, 200, 0), 5)
            cv2.circle(frame, face_center, 4, (0, 200, 0), 3)

        return face_centers, frame

    def show_frame(self,frame):
        cv2.imshow('img', frame)
        k = cv2.waitKey(6) & 0xff

    """def convert_rpy(angles):

        # This is very stupid:
        # For some reason this doesnt work if exactly  one value = 0
        # the following simply make it a very small value if that happens
        # I do not understand the math behind this well enough to create a better solution
        zeros = 0
        zero_pos = None
        for i,ang in enumerate(angles):
            if ang == 0 :
                zeros += 1
                zero_pos = i
        if zeros == 1:
            #logging.debug("rotation value" + str(zero_pos+1) +"is 0 a small value is added")
            angles[zero_pos] = 1e-6

        roll = angles[0]
        pitch = angles[1]
        yaw = angles[2]

        # print ("roll = ", roll)
        # print ("pitch = ", pitch)
        # print ("yaw = ", yaw)
        # print ("")

        for ang in angles:
            # print(ang % np.pi)
            pass

        if roll == pitch == yaw:

            if roll % np.pi == 0:
                rotation_vec = [0, 0, 0]
                return rotation_vec

        yawMatrix = np.matrix([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
        ])
        # print("yawmatrix")
        # print(yawMatrix)

        pitchMatrix = np.matrix([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        # print("pitchmatrix")
        # print(pitchMatrix)

        rollMatrix = np.matrix([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
        ])
        # print("rollmatrix")
        # print(rollMatrix)

        R = yawMatrix * pitchMatrix * rollMatrix
        # print("R")
        # print(R)

        theta = math.acos(((R[0, 0] + R[1, 1] + R[2, 2]) - 1) / 2)
        # print("theta = ",theta)
        multi = 1 / (2 * math.sin(theta))
        # print("multi = ", multi)


        rx = multi * (R[2, 1] - R[1, 2]) * theta
        ry = multi * (R[0, 2] - R[2, 0]) * theta
        rz = multi * (R[1, 0] - R[0, 1]) * theta

        rotation_vec = [rx,ry,rz]
        # print(rx, ry, rz)
        return rotation_vec
    """

    def check_max_xy(self,xy_coord):
        """
        Checks if the face is outside of the predefined maximum values on the lookaraound plane

        Inputs:
            xy_coord: list of 2 values: x and y value of the face in the lookaround plane.
                These values will be evaluated against max_x and max_y

        Return Value:
            x_y: new x and y values
                if the values were within the maximum values (max_x and max_y) these are the same as the input.
                if one or both of the input values were over the maximum, the maximum will be returned instead
        """

        x_y = [0, 0]
        # print("xy before conversion: ", xy_coord)
        max_x = self.max_x
        max_y = self.max_y
        if -max_x <= xy_coord[0] <= max_x:
            # checks if the resulting position would be outside of max_x
            x_y[0] = xy_coord[0]
        elif -max_x > xy_coord[0]:
            x_y[0] = -max_x
        elif max_x < xy_coord[0]:
            x_y[0] = max_x
        else:
            raise Exception(" x is wrong somehow:", xy_coord[0], -max_x, max_x)

        if -max_y <= xy_coord[1] <= max_y:
            # checks if the resulting position would be outside of max_y
            x_y[1] = xy_coord[1]
        elif -max_y > xy_coord[1]:
            x_y[1] = -max_y
        elif max_y < xy_coord[1]:
            x_y[1] = max_y
        else:
            raise Exception(" y is wrong somehow", xy_coord[1], max_y)
        # print("xy after conversion: ", x_y)

        return x_y

    def set_lookorigin(self):
        """
        Creates a new coordinate system at the current robot tcp position.
        This coordinate system is the basis of the face following.
        It describes the midpoint of the plane in which the robot follows faces.

        Return Value:
            orig: math3D Transform Object
                characterises location and rotation of the new coordinate system in reference to the base coordinate system

        """
        position = self.robotUR.get_actual_tcp_pose()
        orig = m3d.Transform(position)
        return orig

    def move_to_face(self,list_of_facepos, robot_pos):
        """
        Function that moves the robot to the position of the face

        Inputs:
            list_of_facepos: a list of face positions captured by the camera, only the first face will be used
            robot_pos: position of the robot in 2D - coordinates

        Return Value:
            prev_robot_pos: 2D robot position the robot will move to. The basis for the next call to this funtion as robot_pos
        """

        face_from_center = list(list_of_facepos[0])  # TODO: find way of making the selected face persistent

        prev_robot_pos = robot_pos
        scaled_face_pos = [c * m_per_pixel for c in face_from_center]

        robot_target_xy = [a + b for a, b in zip(prev_robot_pos, scaled_face_pos)]
        # print("..", robot_target_xy)

        robot_target_xy = self.check_max_xy(robot_target_xy)
        prev_robot_pos = robot_target_xy

        x = robot_target_xy[0]
        y = robot_target_xy[1]
        z = 0
        xyz_coords = m3d.Vector(x, y, z)

        x_pos_perc = x / self.max_x
        y_pos_perc = y / self.max_y

        x_rot = x_pos_perc * self.hor_rot_max
        y_rot = y_pos_perc * self.vert_rot_max * -1

        tcp_rotation_rpy = [y_rot, x_rot, 0]
        # tcp_rotation_rvec = convert_rpy(tcp_rotation_rpy)
        tcp_orient = m3d.Orientation.new_euler(tcp_rotation_rpy, encoding='xyz')
        position_vec_coords = m3d.Transform(tcp_orient, xyz_coords)

        oriented_xyz = self.origin * position_vec_coords
        oriented_xyz_coord = oriented_xyz.get_pose_vector()

        coordinates = oriented_xyz_coord

        #qnear = self.robotUR.get_actual_joint_positions()
        next_pose = coordinates
        self.robotUR.set_realtime_pose(next_pose)

        return prev_robot_pos
    
def check_exhibit_time():
    pass


robot_ip = "10.211.55.5"
robot = Robot(robot_ip)
robot.initialise_robot()
robot.move_home()

robot.current_row = 0
robot.start_rtde()
time.sleep(5)
while True:
    robot.wander()
    face = robot.follow_face(close=True)
    
    #print("face", face)
    landmark_queue = mp.Queue()
    emotion_queue = mp.Queue()

    coordinates = mp.Process(target=robot.create_coordinates, args=(face,))  # evaluates emotion, creates landmarks returns coordinate list of face drawing and emotion text
    movement = mp.Process(target=robot.move_to_write, args=(robot.current_row,))

    coordinates.start()
    movement.start()

    coordinates.join()
    movement.join()

    landmarks = landmark_queue.get()
    emotions = emotion_queue.get()
    robot.draw_face(landmarks)
    robot.write_emotions(emotions)

    robot.check_paper(robot.current_row)

    robot.current_row += 1
    robot.move_home()