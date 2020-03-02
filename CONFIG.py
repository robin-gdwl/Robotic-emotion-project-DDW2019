import math
import sys

RASPBERRY_BOOL = False
if sys.platform == "linux":
    RASPBERRY_BOOL = True
    import picamera
    from picamera.array import PiRGBArray


#ROBOT_IP = "10.211.55.5"
ROBOT_IP = "192.168.178.20"

FACE_ROW_OFFSET = [0, 0.04]
TEXT_HOR_OFFSET = 0.03
Z_HOP = -0.03
DRAWING_ZVAL = 0.016
CURRENT_ROW = 0
MAX_ROWS = 1
LINE_SPACING = 0.015
BLEND_RADIUS = 0.0002
TEXT_SCALING = 1/120
LETTER_SPACING = 0.005

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
PRINT_COORDINATES = True

# self.face_detect = dlib.get_frontal_face_detector()
#dnnFaceDetector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
# paper advancing
DRAG_DIST = 0.10  # 10 cm
#PLUNGE_DIST = 0.1273
PLUNGE_DIST = 0.06
PAPERSLOT_START = [0.02, -0.548, 0.1980, 0.0, -3.14, 0]

# positions
HOME_POS = (math.radians(50),
                 math.radians(-63),
                 math.radians(-93),
                 math.radians(-20),
                 math.radians(88),
                 math.radians(0))

ABOVE_PAPER = (math.radians(-73),
                 math.radians(-105.5),
                 math.radians(-116),
                 math.radians(-46.4),
                 math.radians(89.5),
                 math.radians(17))

PROGRAMSTATE = 0 
ROBOT_ACTION = 0


VIDEO_RESOLUTION = (700, 400)  # resolution the video capture will be resized to, smaller sizes can speed up detection
VIDEO_MIDPOINT = (int(VIDEO_RESOLUTION[0]/2),
                  int(VIDEO_RESOLUTION[1]/2))
VIDEO_ASPECT_RATIO  = VIDEO_RESOLUTION[0] / VIDEO_RESOLUTION[1]  # Aspect ration of each frame
VIDEO_VIEWANGLE_HOR = math.radians(25)  # Camera FOV (field of fiew) angle in radians in horizontal direction
#video_viewangle_vert = video_viewangle_hor / video_asp_ratio  #  Camera FOV (field of fiew) angle in radians in vertical direction
M_PER_PIXEL = 00.00006  # Variable which scales the robot movement from pixels to meters.