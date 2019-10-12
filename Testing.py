import time
import random
#from Robot_Motion import RobotMotion
from Coordinate_conversion import Coord, RobotCoord, ScreenCoord
from Face_Detection_Operations import FaceOperation # not yet fully written

# THis file is there to test Classes and functions without modifying Main_program.py
# It is not needed to run the Main Program.
# THis file might get a bit messy...



# Test facefinder :

face_finder = FaceOperation()

while True:
    #print(face_finder.findface())
    if face_finder.findface() == False:
        print ("no face")
    else:
        print("face detected")
        list_facepos = face_finder.facelocation()
        print("list face pos: ", list_facepos)
        face_screen_location = ScreenCoord(list_facepos[0], list_facepos[1], 1, 1)
        face_real_location = face_screen_location.convert_screen_coords()
        #print(face_screen_location)
        print(face_real_location)
