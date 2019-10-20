import time
from math import pi,sqrt
import urx
import math3d as m3d


# simple class that just slightly extends the urx capabilites
# combines some of the urx functions for convenience and easy calling in the main Program
# !: remember that urx uses meters as units

class RobotMotion:

    def __init__(self):
        #alternate IP: "192.168.178.20"
        # 10.210.155.126
        #self.IP = "172.23.4.26"
        self.IP = "192.168.178.20"
        self.a = 0.3
        self.v = 0.3
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
            time.sleep(1)
            self.connect()

    def lookaround(self):
        return []


    def move_home(self):
        self.move_between()
        # self.robot.movej((1.841301679611206, -1.6310561339007776, -1.7878111044513147, 0.28027474880218506, 1.30446195602417, 0), self.a, self.v)
        #self.robot.movej((-0.028123203908101857, -1.1558621565448206, -1.9442971388446253, -0.012953106557027638, 1.5486491918563843, 0.011556537821888924), self.a, self.v)
        self.robot.movej((1.7691065073013306, -1.0238812605487269, -2.190423313771383, 0.09588956832885742, 1.3761399984359741, 0.052045244723558426), self.a, self.v)


        print("Robot moved to home position.")
        self.robot.csys = m3d.Transform() # reset csys otherwise weird things happen.
        home_csys = self.robot.get_pose()
        self.robot.set_csys(home_csys)
        print("Csys set to current pose: Look")

    def test_move(self): # this should test weather the move will stay inside of the lookarea


        return None

    def move(self, full_coords): # gets a list of 6 values and moves the robot according to these values
        # print("full coords: ", full_coords)
        self.robot.movel(full_coords, self.a*0.5, self.v*0.5)

        return None

    def move_between(self):
        self.robot.movej((0.5004713535308838, -0.884106461201803, -1.5667465368853968, -0.792891804371969, 1.5332516431808472, 0.1559387445449829), self.a, self.v)
        #self.robot.movej((-1.198425594960348, -1.518754784260885, -1.8426645437823694, -0.7939837614642542, 1.5331677198410034, 0.15597468614578247), self.a, self.v)

    def move_to_write(self):

        self.move_between()
        self.robot.movej((-1.198425594960348, -1.518754784260885, -1.8426645437823694, -0.7939837614642542,
                          1.5331677198410034, 0.15597468614578247), self.a, self.v *0.8)

        # über dem papier 01 self.robot.movej((-1.186561409627096, -1.9445274511920374, -1.7661479155169886, -1.006078068410055, 1.5503629446029663, 0.3756316900253296), self.a, self.v)
        self.robot.movej((-1.2749927679644983, -1.9379289785968226, -2.09098464647402, -0.6840408484088343, 1.5629680156707764, 0.28495118021965027), self.a, self.v)
        # if movel should be used csys has to be reset to base before move
        # #self.robot.movel((0.6000000521429313, 0.15000001308942848, 0.09000014933946439, -2.221440281881211, -2.2214404854075736, -1.74503212199567e-07), self.a, self.v)

        self.robot.csys = m3d.Transform() # reset csys otherwise weird things happen...
        write_csys = self.robot.get_pose()

        time.sleep(0.1)
        self.robot.set_csys(write_csys)
        time.sleep(0.3)
        #print("write_csys: ", write_csys)
        print("Csys set to current pose: Write")
        return None

    def draw_landmarks(self, landmark_coords):
        a = 0.2
        s = 0.8
        # print("landmark coords:",landmark_coords)
        for coord in landmark_coords:
            coord.extend([0, 0, 0])
            # print(coord)
        time.sleep(1)
        self.robot.movels(landmark_coords, a, s, 0.00015)

        return None

    def write_results(self, results):

        a = 0.3
        s = 0.9

        result_as_coords = []  # python y u no work? this seems unecessary but i cant get it to work otherwise

        for coord in results:
            coord = coord.copy()
            coord.extend([0, 0, 0])
            #print("coord: ", coord)
            result_as_coords.append(coord.copy())
        time.sleep(1)

        #print("results as coordinate list: ", result_as_coords)
        self.robot.movels(result_as_coords, a, s, 0.00015)



    def move_paper(self):
        drag_dist = 0.10
        plunge_dist = 0.1269

        print("moving paper")

        self.robot.set_csys(m3d.Transform())  # reset csys otherwise weird things happen...
        self.robot.movel((0.013,       -0.548,         0.1980, 0.0, -3.14, 0), self.a, self.v)
        self.robot.movel((0.0,            0.0,  - plunge_dist, 0.0, 0, 0), self.a, self.v, relative=True)
        self.robot.movel((- drag_dist,    0.0,            0.0, 0.0, 0, 0), self.a, self.v, relative=True)
        self.robot.movel((0.0,            0.0,    plunge_dist, 0.0, 0, 0), self.a, self.v, relative=True)
        self.robot.movel((0.0,            0.0, -plunge_dist/2, 0.0, 0, 0), self.a*2.5, self.v*2.6, relative=True)
        time.sleep(3)
        '''self.robot.movels([(0.009, -0.578, 0.280, 0.794, -3.14, 0),
                           (0.009, -0.578, 0.080, 0.794, -3.14, 0),
                           (-0.1, -0.578, 0.080, 0.794, -3.14, 0),
                           (-0.1, -0.578, 0.280, 0.794, -3.14, 0)
                           ],self.a, self.v)
                           '''
        print("paper moved, hopefully it fell off the pen again...  ")

        return None

    def pause(self):
        # get current Position
        # if position = rest position:
            # do nothing
        # else:
            # go to rest position via highpoint
        return None

