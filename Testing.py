import time
import random
#from Robot_Motion import RobotMotion
from Coordinate_conversion import Coord, RobotCoord, ScreenCoord
from Face_Detection_Operations import FaceOperation # not yet fully written

# THis file is there to test Classes and functions without modifying Main_program.py
# It is not needed to run the Main Program.
# THis file might get a bit messy...

# TODO: test robot lookaround


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

        print(face_real_location)

# Test Robot lookaround

Robot = RobotMotion()
face_finder = FaceOperation()
lookarea_x = 0.4    # overall x- extent of the (rectangular) area in which the robot looks around
lookarea_y = 0.4    # overall y- extent of the (rectangular) area in which the robot looks around

while True
    if face_finder.findface == False:
        print("No one around. Maybe over here? ")
        # Generate a random xy-coordinate in the robot look area:
        look_x = random.uniform(- (lookarea_x / 2), (lookarea_x / 2))
        look_y = random.uniform(- (lookarea_y / 2), (lookarea_y / 2))
        coordinates = RobotCoord(look_x, look_y, lookarea_x,
                                 lookarea_y)  # creates a RobotCoords object with the random xy coordinates
        full_coords = coordinates.convert_robot_coords()  # converts the object to a full 6D coordinate
        Robot.move(full_coords)  # move the robot to the random coordinates with the correct z and rotation

    else:
        print("face detected, not looking around anymore")
