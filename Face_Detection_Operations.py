import dlib
import cv2
import tensorflow
import time

class FaceOperation:

    # there is no visualisation of this process yet
    # the camera activates but all image processing is done in the background
    # preview with cv2.imshow("Frame", frame)

    def __init__(self):

        self.landmarks = []
        self.emotion = []
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
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
        # TODO: combine this with findface() to not do the same operations twice

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
        _, frame = self.cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = self.detector(gray)
        # TODO: this shouldn't be called in each method. make it a parmeter, check if it is there, redo if necessary
        if len(faces) >= 1:
            face = faces[0]
            landmarks = self.predictor(gray, face)
            landmark_points = []
            face_scale = 1  # this will be used to scale each face to a similar size
            face_target_size = 200

            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # scale the face coordinates:
            face_width = face.right() - face.left()
            face_height = face.bottom() - face.top()
            face_average = (face_width + face_height) / 2
            print("face width and height: ", face_width, " , ", face_height)
            print("face average size: ", face_average)
            face_scale = face_target_size / face_average


            for n in range(0, 68):
                #print(landmarks.part(n))
                x = int((landmarks.part(n).x - face.left()) * face_scale)
                y = int((landmarks.part(n).y - face.top()) * face_scale)
                cv2.circle(frame, (x, y), 3, (100, 0, 0), -1)

                x = landmarks.part(n).x * face_scale - face.left() # apply scale and move to upper left corner
                y = landmarks.part(n).y * face_scale - face.top() # as above
                landmark_points.append([x, y])



            # individual features, each of these will be one continuous line to be drawn
            jawline = landmark_points[:17]
            left_brow = landmark_points[17:22]
            right_brow = landmark_points[22:27]
            nose_ridge = landmark_points[27:31]
            nose_tip = landmark_points[31:36]
            left_eye = landmark_points[36:42].append(landmark_points[36]) # add the first point to get a closed curve
            right_eye = landmark_points[42:48].append(landmark_points[42])
            lips_outer = landmark_points[48:61].append(landmark_points[48])
            lips_inner = landmark_points[61:68].append(landmark_points[61])

            feature_lines = [jawline, left_brow, right_brow, nose_ridge, nose_tip, left_eye, right_eye, lips_outer, lips_inner]
            # feature_lines is now a list of all lines to be drawn
            #cv2.imshow("Frame", frame)
            #cv2.waitKey(1)

            return feature_lines

    def apply_zhop(self, list_of_lines, z_hop = 0.05):

        for line in list_of_lines:

            lines_w_zhop = 0


    def detect_emotion(self):
        # detects the emotion and returns a list of three strings: ["most common emotion: 20%", ....]
        return None