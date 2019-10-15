import time
import random
from Robot_Motion import RobotMotion
from Coordinate_conversion import Coord, RobotCoord, ScreenCoord
from Face_Detection_Operations import FaceOperation # not yet written
from String_to_Path import ThingToWrite


Robot = RobotMotion()
face_finder = FaceOperation()
lookarea_x = 0.4    # overall x- extent of the (rectangular) area in which the robot looks around
lookarea_y = 0.4    # overall y- extent of the (rectangular) area in which the robot looks around


Robot.move_home()

while True:  # This is the actual process: lookaround then face tracking if a face is found and lastly write and draw
    while face_finder.findface() == False:
        print("No one around. Maybe over here? ")
        # Generate a random xy-coordinate in the robot look area:
        look_x = random.uniform(- (lookarea_x/2), (lookarea_x/2))
        look_y = random.uniform(- (lookarea_y / 2), (lookarea_y / 2))
        coordinates = RobotCoord(look_x, look_y, lookarea_x, lookarea_y) # creates a RobotCoords object with the random xy coordinates
        full_coords = coordinates.convert_robot_coords()  # converts the object to a full 6D coordinate
        Robot.move(full_coords)  # move the robot to the random coordinates with the correct z and rotation

    watch_time = time.time() + 5

    # TODO: implement inverse kinematics and use a speedl or servoc
    while True:
        # this loop is used to track the face. If no face is detected the loop breaks and the robot looks around instead (see above)
        # if a face is detected it is tracked and after a time its analysed and recorded

        if face_finder.findface() == True:

            list_facepos = face_finder.facelocation()
            print("list face pos: ", list_facepos)
            robot_pos = Robot.robot.getl()
            screensize = [face_finder.screen_width, face_finder.screen_height]
            face_screen_location = ScreenCoord(list_facepos[0], list_facepos[1], lookarea_x, lookarea_y, robot_pos, screensize)
            face_real_location = face_screen_location.convert_screen_coords()
        else:
            break

        #Shorten the move if it would go outside of the lookarea:
        if (-lookarea_x / 2) <= face_real_location[0] <= (lookarea_x / 2):  #
            pass
        else:
            print("outside of x-extent")
            if face_real_location[0] > (lookarea_x / 2):
                face_real_location[0] = (lookarea_x / 2)
            else:
                face_real_location[0] = (-lookarea_x / 2)

        if (-lookarea_y / 2) <= face_real_location[1] <= (lookarea_y / 2):
            pass
        else:
            print("outside of y-extent")
            if face_real_location[1] > (lookarea_x / 2):  # If
                face_real_location[1] = (lookarea_x / 2)
            else:
                face_real_location[1] = (-lookarea_x / 2)

        if time.time() < watch_time:  # only start writing after looking at a face for a certain time
            Robot.move(face_real_location)
            continue

        else:  # if the watch_time has passed the actual face evaluation begins
            face_finder.landmark_detection()
            face_landmarks = face_finder.landmarks  # should be a list of list of coordinates
            face_finder.detect_emotion()
            emotion_score = face_finder.emotion  # list of strings with top 3 emotions
            for string in emotion_score:
                string = ThingToWrite(string)
                converted_string = string.string_to_coordinates()

            # write the results of the evaluation
            Robot.move_to_write()
            print("current l: ", Robot.robot.getl())
            Robot.draw_landmarks(face_landmarks)
            emotions = face_finder.detect_emotion()
            print(emotions)
            for emotion in emotions:
                emotion_coords = ThingToWrite(emotion).string_to_coordinates() # add origin here
                Robot.write_results(emotion_coords)
            # ADD: test wether to advance the paper roll
            #Robot.move_paper()
            Robot.move_home()

            break
