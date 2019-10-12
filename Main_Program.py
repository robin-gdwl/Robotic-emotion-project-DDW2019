import time
import random
from Robot_Motion import RobotMotion
from Coordinate_conversion import Coord, RobotCoord, ScreenCoord
from Face_Detection_Operations import FaceOperation # not yet written



Robot = RobotMotion()
face_finder = FaceOperation()
lookarea_x = 0.4    # overall x- extent of the (rectangular) area in which the robot looks around
lookarea_y = 0.4    # overall y- extent of the (rectangular) area in which the robot looks around

Robot.move_home()

while True:  # maybe put the frame capture in here instead?
    while face_finder.findface == False:
        print("No one around. Maybe over here? ")
        # Generate a random xy-coordinate in the robot look area:
        look_x = random.uniform(- (lookarea_x/2), (lookarea_x/2))
        look_y = random.uniform(- (lookarea_y / 2), (lookarea_y / 2))
        coordinates = RobotCoord(look_x, look_y, lookarea_x, lookarea_y) # creates a RobotCoords object with the random xy coordinates
        full_coords = coordinates.convert_robot_coords()  # converts the object to a full 6D coordinate
        Robot.move(full_coords)  # move the robot to the random coordinates with the correct z and rotation

    watch_time = time.time() + 5

    while True:
        if face_finder.findface() == True:

            list_facepos = face_finder.facelocation()
            print("list face pos: ", list_facepos)
            face_screen_location = ScreenCoord(list_facepos[0], list_facepos[1], 1, 1)
            face_real_location = face_screen_location.convert_screen_coords()
        else:
            break

        in_bounds = Robot.test_move()

        if time.time() < watch_time and in_bounds ==True:
            Robot.move(face_real_location)
            continue

        else:
            face_finder.landmark_detection()
            face_landmarks = face_finder.landmarks  #should be a list of list of coordinates
            face_finder.detect_emotion()
            emotion_score = face_finder.emotion  # list of strings with top 3 emotions

            Robot.move_to_write()
            Robot.draw_landmarks(face_landmarks)
            Robot.write_results(emotion_score)
            Robot.move_home()

            break
