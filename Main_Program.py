import time
import datetime
import random
import threading
import numpy as np
from Robot_Motion import RobotMotion
from Coordinate_conversion import Coord, RobotCoord, ScreenCoord
from Face_Detection_Operations import FaceOperation 
from String_to_Path import ThingToWrite

# TODO: Refactor this mess of a code

Robot = RobotMotion()
face_finder = FaceOperation()

lookarea_x = 0.43    # overall x- extent of the (rectangular) area in which the robot looks around
lookarea_y = 0.3    # overall y- extent of the (rectangular) area in which the robot looks around
position = 0
iteration = 0
draw_origin = [0,0]
origin_offset = [0.05, 0.01]
line_spacing = 0.01

exhibit_start_hr = 9
exhibit_end_hr = 18

message_after = 28  # number of evaluations after which the proj_message is written
proj_message = ["______________________",
                "HOW DO I SEE TECHNOLOGY",
                "WHEN I REALISE ",
                "TECHNOLOGY SEES ME",
                " - ? -",
                "______________________",
                "Robotic emotion project",
                "created by:",
                "Robin Godwyll and Yang N i"]  # This message is written by the robot after a certain amount of evaluations

Robot.move_home()
out_of_time_bool = False

while True:  # This is the actual process: lookaround then face tracking if a face is found and lastly write and draw
    curr_time = datetime.datetime.now()
    print("Date and time:", curr_time)
    curr_time = curr_time.hour

    if exhibit_start_hr <= curr_time < exhibit_end_hr:  # disabled because the time on the raspberry pi is not consistent

        if out_of_time_bool == True:
            Robot.move_home()
            out_of_time_bool = False

        face_finder.getframe()  # take image with pi camera
        while face_finder.findface() == False and exhibit_start_hr <= curr_time < exhibit_end_hr:
            print("No one around. Maybe over here? ")
            # Generate a random xy-coordinate in the robot look area:
            look_x = random.uniform(- (lookarea_x/3), (lookarea_x/3))
            look_y = random.uniform(- (lookarea_y / 3), (lookarea_y / 3))
            coordinates = RobotCoord(look_x, look_y, lookarea_x, lookarea_y) # creates a RobotCoords object with the random xy coordinates
            full_coords = coordinates.convert_robot_coords()  # converts the object to a full 6D coordinate
            Robot.move(full_coords)  # move the robot to the random coordinates with the correct z and rotation
            face_finder.getframe()  # take new image after moving

            curr_time = datetime.datetime.now()
            print("Date and time:", curr_time)
            curr_time = curr_time.hour

        watch_time = time.time() + 6

        # TODO: implement inverse kinematics and use a speedl or servoc to make face tracking more reliable

        while True:
            # this loop is used to track the face. If no face is detected the loop breaks and the robot looks around instead (see above)
            # if a face is detected it is tracked and after a time its analysed and recorded

            if face_finder.facelocation() == True:

                list_facepos = face_finder.face_loc
                print("list face pos: ", list_facepos)
                robot_pos = Robot.robot.getl()
                screensize = [1080, 720]
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
                Robot.move(face_real_location, thresh= 0.1)
                face_finder.getframe()
                continue

            else:  # if the watch_time has passed the actual face evaluation begins
                iteration += 1
                print("Face evaluation number: ", iteration)

                if position == 0:
                    draw_origin = [0.00, -0.04]
                elif position == 1:
                    draw_origin = [0.00, 0.015]
                elif position == 2:
                    draw_origin = [0.0, 0.07]
                else:
                    draw_origin = [0.0, 0.125]

                face_finder.landmark_detection(draw_origin)
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

                draw_origin[0] += origin_offset[0]
                draw_origin[1] += origin_offset[1]

                for emotion in emotions:
                    emotion_coords = ThingToWrite(emotion).string_to_coordinates(draw_origin) # add origin here
                    Robot.write_results(emotion_coords)
                    draw_origin[1] += line_spacing

                if position == 3:
                    position = 0
                    print("position reset, iteration:",iteration)
                    Robot.move_paper()  # move the paper after drawing 4 faces
                else:
                    position += 1

                if iteration >= message_after  and position == 0:
                    draw_origin = [0,0]
                    Robot.move_to_write()
                    for line in proj_message:
                        print("writing: ", line)
                        message_coords = ThingToWrite(line).string_to_coordinates(draw_origin)  # add origin here
                        Robot.write_results(message_coords)
                        draw_origin[1] += line_spacing + 0.002
                    iteration = 0
                    Robot.move_paper()
                    time.sleep(30)

                Robot.move_home()

                break

    else:  #  stop robot when the exhibition is closed
        out_of_time_bool = True
        print(datetime.datetime.now())
        print("outside of exhibition hours - Robot stopped")
        Robot.fold()
        time.sleep(240)


