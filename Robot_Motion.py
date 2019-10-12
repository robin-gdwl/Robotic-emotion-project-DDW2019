import time
from math import pi,sqrt
import urx


# simple class that just slightly extends the urx capabilites
# combines some of the urx functions for convenience and easy calling in the main Program
# !: remember that the urx uses meters as units

class RobotMotion:

    def __init__(self):
        self.robot = urx.Robot("192.168.178.22")
        self.a = 0.3
        self.v = 0.1
        self.csys_look = []
        self.csys_write = []
        print("robot initiated")

    def lookaround(self):
        return []

    def move_home(self):
        self.robot.movej((1.841301679611206, -1.6310561339007776, -1.7878111044513147, 0.28027474880218506, 1.30446195602417, 0), self.a, self.v)
        print("Robot moved to home position.")
        home_csys = self.robot.get_pose()
        self.robot.set_csys(home_csys)
        print("Csys set to current pose.")

    def test_move(self): # this should test weather the move will stay inside of the lookarea


        return None

    def move(self, full_coords): # gets a list of 6 values and moves the robot according to these values
        # print("full coords: ", full_coords)
        self.robot.movel(full_coords, self.a, self.v)

        return None

    def move_to_write(self):
        return None

    def draw_landmarks(self, landmarks):
        return None

    def write_results(self, results):
        return None

