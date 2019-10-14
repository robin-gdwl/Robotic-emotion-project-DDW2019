import time
from math import pi,sqrt
import urx
import math3d as m3d


# simple class that just slightly extends the urx capabilites
# combines some of the urx functions for convenience and easy calling in the main Program
# !: remember that urx uses meters as units

class RobotMotion:

    def __init__(self):
        #self.robot = urx.Robot("192.168.178.22")
        self.robot = urx.Robot("192.168.178.22")
        self.a = 0.3
        self.v = 0.6
        #self.csys_look = []  # not yet used anywhere
        #self.csys_write = []  # not yet used anywhere
        print("robot initiated")

    def lookaround(self):
        return []

    def move_home(self):
        self.robot.movej((1.841301679611206, -1.6310561339007776, -1.7878111044513147, 0.28027474880218506, 1.30446195602417, 0), self.a, self.v)
        print("Robot moved to home position.")
        home_csys = self.robot.get_pose()
        self.robot.csys = m3d.Transform() # reset csys otherwise weird things happen.
        self.robot.set_csys(home_csys)
        print("Csys set to current pose: Look")

    def test_move(self): # this should test weather the move will stay inside of the lookarea


        return None

    def move(self, full_coords): # gets a list of 6 values and moves the robot according to these values
        # print("full coords: ", full_coords)
        self.robot.movel(full_coords, self.a, self.v, threshold=200)

        return None

    def move_to_write(self):

        #self.robot.movej(
        #    (0.4223927855491638, -2.148930374776022, -1.76170522371401, -0.8017538229571741, 1.5707978010177612, 3.5639853477478027),
        #    self.a, self.v)

        self.robot.movej(
            (1.5707963268, -2.097018543873922, -1.7439802328692835, -0.8702052275287073, 1.5746145248413086,.5161566734313965),
            self.a, self.v)

        # if movel should be used csys has to be reset to base before move
        # #self.robot.movel((0.6000000521429313, 0.15000001308942848, 0.09000014933946439, -2.221440281881211, -2.2214404854075736, -1.74503212199567e-07), self.a, self.v)

        self.robot.csys = m3d.Transform() # reset csys otherwise weird things happen...
        write_csys = self.robot.get_pose()

        time.sleep(0.1)
        self.robot.set_csys(write_csys)
        time.sleep(0.3)
        print("write_csys: ", write_csys)
        print("Csys set to current pose: Write")
        return None

    def draw_landmarks(self, landmark_coords):
        print("landmark coords:",landmark_coords)
        for coord in landmark_coords:
            coord.extend([0, 0, 0])
            print(coord)
        time.sleep(1)
        self.robot.movels(landmark_coords, self.a, self.v, 0.0015)

        return None

    def write_results(self, results= [["unknown1",0],["unknown2",0],["unknown3",0]]):

        for result in results:
            percentage = result[1]
            emotion = result[0]
            print(f"{percentage} % {emotion}")
            return None

    def move_paper(self):
        return None