import time
from math import pi,sqrt
import urx
import math3d as m3d


# simple class that just slightly extends the urx capabilites
# combines some of the urx functions for convenience and easy calling in the main Program
# !: remember that urx uses meters as units

class RobotMotion:

    def __init__(self):
        #alternate IP: "192.168.178.22"
        #self.IP = "172.23.4.26"
        self.IP = "192.168.178.22"
        self.a = 0.3
        self.v = 0.4
        #self.csys_look = []  # not yet used anywhere
        #self.csys_write = []  # not yet used anywhere
        self.robot = None
        print("robot initiated")
        self.connect()



    def connect(self):
        try:
            self.robot = urx.Robot(self.IP)
        except:
            print("retrying to connect to robot")
            self.connect()

    def lookaround(self):
        return []


    def move_home(self):
        # self.robot.movej((1.841301679611206, -1.6310561339007776, -1.7878111044513147, 0.28027474880218506, 1.30446195602417, 0), self.a, self.v)
        self.robot.movej((-0.028123203908101857, -1.1558621565448206, -1.9442971388446253, -0.012953106557027638, 1.5486491918563843, 0.011556537821888924), self.a, self.v)

        print("Robot moved to home position.")
        self.robot.csys = m3d.Transform() # reset csys otherwise weird things happen.
        home_csys = self.robot.get_pose()
        self.robot.set_csys(home_csys)
        print("Csys set to current pose: Look")

    def test_move(self): # this should test weather the move will stay inside of the lookarea


        return None

    def move(self, full_coords): # gets a list of 6 values and moves the robot according to these values
        # print("full coords: ", full_coords)
        self.robot.movel(full_coords, self.a, self.v, threshold=200)

        return None

    def move_to_write(self):

        self.robot.movej((-1.188862148915426, -1.9500582853900355, -1.8263033072101038, -0.9351370970355433, 1.5643458366394043, 0.37276408076286316), self.a, self.v)

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

    def write_results(self, results):

        result_as_coords = []  # python y u no work? this seems unecessary but i cant get it to work otherwise

        for coord in results:
            coord = coord.copy()
            coord.extend([0, 0, 0])
            print("coord: ", coord)
            result_as_coords.append(coord.copy())
        time.sleep(1)

        print("results as coordinate list: ", result_as_coords)
        self.robot.movels(result_as_coords, self.a, self.v, 0.00015)



    def move_paper(self):
        return None