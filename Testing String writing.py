import time
import random
from Robot_Motion import RobotMotion
from Coordinate_conversion import Coord, RobotCoord, ScreenCoord
from Face_Detection_Operations import FaceOperation # not yet fully written
from String_to_Path import ThingToWrite

Robot = RobotMotion()

Robot.move_to_write()
print("current l: ", Robot.robot.getl())
#Robot.draw_landmarks(face_landmarks)
emotions = ["--", "hallo leute ihr seid cool", "33% happy", "45% surprisedd"]
print(emotions)
for emotion in emotions:
    emotion_coords = ThingToWrite(emotion).string_to_coordinates() # add origin here
    print("emotion coords: ", emotion_coords)
    Robot.write_results(emotion_coords)