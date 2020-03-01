import math

#ROBOT_IP = "10.211.55.5"
ROBOT_IP = "192.168.178.20"

FACE_ROW_OFFSET = [0, 0.04]
TEXT_HOR_OFFSET = 0.03
Z_HOP = -0.03
DRAWING_ZVAL = 0.01
CURRENT_ROW = 0
MAX_ROWS = 1
LINE_SPACING = 0.01
BLEND_RADIUS = 0.0001

MAX_X = 0.2
MAX_Y = 0.2
HOR_ROT_MAX = math.radians(50)
VERT_ROT_MAX = math.radians(25)
ACCEL = 0.5
VEL= 0.5


FOLLOW_TIME = 50
WANDER_DIST = 0.005
W_ANGLECHANGE = 5.0
ESCAPE_ANGLECHANGE = 45

# reporting
PRINT_COORDINATES = False

# self.face_detect = dlib.get_frontal_face_detector()
#dnnFaceDetector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
# paper advancing
DRAG_DIST = 0.10  # 10 cm
PLUNGE_DIST = 0.1273
PAPERSLOT_START = [0.02, -0.548, 0.1980, 0.0, -3.14, 0]

# positions
HOME_POS = (math.radians(-218),
                 math.radians(-63),
                 math.radians(-93),
                 math.radians(-20),
                 math.radians(88),
                 math.radians(0))

PROGRAMSTATE = 0 
ROBOT_ACTION = 0