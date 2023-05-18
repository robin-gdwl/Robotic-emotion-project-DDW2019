import math
import sys
from PROGRAMSTATE import ProgramState

RASPBERRY_BOOL = False
if sys.platform == "linux":
    RASPBERRY_BOOL = True
    import picamera
    from picamera.array import PiRGBArray

PROGRAMSTATE = ProgramState() 
ROBOT_ACTION = 0

#ROBOT_IP = "10.211.55.5"
ROBOT_IP = "192.168.178.45"

FACE_ACTIVATE = True
FACE_ROW_OFFSET = [0, 0.01]
TEXT_HOR_OFFSET = 0.04
Z_HOP = 0.006
DRAWING_ZVAL = 0.1017
CURRENT_ROW = 6
MAX_ROWS = 6
ROW_SPACING = 0.06
LINE_SPACING = 0.012
BLEND_RADIUS = 0.0005
TEXT_SCALING = 1/120
LETTER_SPACING = 0.006

MAX_X = 0.25
MAX_Y = 0.15
HOR_ROT_MAX = math.radians(45)
VERT_ROT_MAX = math.radians(12)
ACCEL = 0.2
VEL= 0.93


FOLLOW_TIME = 10
WANDER_DIST = 0.003
W_ANGLECHANGE = 5.0
ESCAPE_ANGLECHANGE = 45
MAX_MOTION_ANGLECHANGE = 170

# reporting
PRINT_COORDINATES = False
SHOW_FRAME = True

# self.face_detect = dlib.get_frontal_face_detector()
#dnnFaceDetector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
# paper advancing
DRAG_DIST = 0.05 # 10 cm
#PLUNGE_DIST = 0.1273
PLUNGE_DIST = 0.079
PAPERSLOT_START = [-0.021, -0.502, 0.450, 
                   -0.1136, 3.1213, -0.1356]

# positions
HOME_POS = (math.radians(78),
                 math.radians(-90),
                 math.radians(-85),
                 math.radians(-10),
                 math.radians(102),
                 math.radians(0))

BETWEEN = (math.radians(0),
                 math.radians(-47),
                 math.radians(-125),
                 math.radians(-10),
                 math.radians(102),
                 math.radians(0))

ABOVE_PAPER = (math.radians(-80),
                 math.radians(-90),
                 math.radians(-85),
                 math.radians(-90),
                 math.radians(90),
                 math.radians(6))




VIDEO_RESOLUTION = (640, 480)  # resolution the video capture will be resized to, smaller sizes can speed up detection
VIDEO_MIDPOINT = (int(VIDEO_RESOLUTION[0]/2),
                  int(VIDEO_RESOLUTION[1]/2))
VIDEO_ASPECT_RATIO  = VIDEO_RESOLUTION[0] / VIDEO_RESOLUTION[1]  # Aspect ration of each frame
VIDEO_VIEWANGLE_HOR = math.radians(25)  # Camera FOV (field of fiew) angle in radians in horizontal direction
#video_viewangle_vert = video_viewangle_hor / video_asp_ratio  #  Camera FOV (field of fiew) angle in radians in vertical direction
M_PER_PIXEL = 00.00007  # Variable which scales the robot movement from pixels to meters.

# LED Pins: 
# Led colour pins
