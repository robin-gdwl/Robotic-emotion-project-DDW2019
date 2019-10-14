import time
import random
from Robot_Motion import RobotMotion
from Coordinate_conversion import Coord, RobotCoord, ScreenCoord
from Face_Detection_Operations import FaceOperation # not yet fully written

# THis file is there to test Classes and functions without modifying Main_program.py
# It is not needed to run the Main Program.

'''
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
'''

# Test Robot lookaround
# works as of 12.10. 21:32


#Robot = RobotMotion()
#Robot.move_home()

#face_finder = FaceOperation()
lookarea_x = 0.3    # overall x- extent of the (rectangular) area in which the robot looks around
lookarea_y = 0.3    # overall y- extent of the (rectangular) area in which the robot looks around
'''
while True:
    if face_finder.findface() == False:
        print("No one around. Maybe over here? ")
        # Generate a random xy-coordinate in the robot look area:
        look_x = random.uniform(- (lookarea_x / 2), (lookarea_x / 2))
        look_y = random.uniform(- (lookarea_y / 2), (lookarea_y / 2))
        print("random coordinates: ", look_x, "|", look_y)
        coordinates = RobotCoord(look_x, look_y, lookarea_x, lookarea_y)  # creates a RobotCoords object with the random xy coordinates
        full_coords = coordinates.convert_robot_coords()  # converts the object to a full 6D coordinate
        print(full_coords)
        Robot.move(full_coords)  # move the robot to the random coordinates with the correct z and rotation
        time.sleep(0.1)

    else:
        print("face detected, not looking around anymore")
        time.sleep(2)

'''

'''
# Test Robot test_move
# Test xy - Robot coordinates

coords_to_test = [-3, 2, 4, 0, 0, 0]
amended_coords = [0, 0] # these will be the closest coordinates inside the lookarea

if (-lookarea_x / 2) <= coords_to_test[0] <= (lookarea_x / 2): #
    pass
else:
    print("outside of x-extent")
    if coords_to_test[0] > (lookarea_x / 2):
        amended_coords[0] = (lookarea_x / 2)
    else:
        amended_coords[0] = (-lookarea_x / 2)


if (-lookarea_y / 2) <= coords_to_test[0] <= (lookarea_y / 2):
    pass
else:
    print("outside of y-extent")
    if coords_to_test[1] > (lookarea_x / 2):  # If
        amended_coords[1] = (lookarea_x / 2)
    else:
        amended_coords[1] = (-lookarea_x / 2)
'''


'''
Robot = RobotMotion()
Robot.move_home()
position = Robot.robot.getl()
print(position)
screen_dim = [face_finder.screen_width, face_finder.screen_height]
print("SCREEN dim: ", screen_dim)

# Test Face_tracking:
while True:
    if face_finder.findface() == True:
        list_facepos = face_finder.facelocation()
        #print("list face pos: ", list_facepos)
        face_screen_location = ScreenCoord(list_facepos[0], list_facepos[1], lookarea_x, lookarea_y, Robot.robot.getl(), screen_dim)
        face_real_location = face_screen_location.convert_screen_coords()

        if (-lookarea_x / 2) <= face_real_location[0] <= (lookarea_x / 2):  #
            pass
        else:
            #print("outside of x-extent")
            if face_real_location[0] > (lookarea_x / 2):
                face_real_location[0] = (lookarea_x / 2)
            else:
                face_real_location[0] = (-lookarea_x / 2)

        if (-lookarea_y / 2) <= face_real_location[1] <= (lookarea_y / 2):
            pass
        else:
            #print("outside of y-extent")
            if face_real_location[1] > (lookarea_x / 2):  # If
                face_real_location[1] = (lookarea_x / 2)
            else:
                face_real_location[1] = (-lookarea_x / 2)

        Robot.move(face_real_location)
'''

print("lets go")
face_finder = FaceOperation()
time.sleep(1)
face_finder.landmark_detection()
face_landmarks = face_finder.landmarks
print("face lm: ", face_landmarks)


Robot = RobotMotion()
#print(Robot.robot.getl())
print(Robot.robot.getj())

#Robot.move_home()
Robot.move_to_write()
Robot.draw_landmarks(face_landmarks)
#Robot.write_results(emotion_score)
#[0.6000000521429313, 0.15000001308942848, 0.09000014933946439, -2.221440281881211, -2.2214404854075736, -1.74503212199567e-07]
#[0.4223927855491638, -2.148930374776022, -1.76170522371401, -0.8017538229571741, 1.5707978010177612, 3.5639853477478027]

results = [["unknown1", 0], ["unknown2", 0], ["unknown3", 0]]

for result in results:
    percentage = result[1]
    emotion = result[0]
    print(f"{percentage} %  {emotion}")

#where to get the emotion string