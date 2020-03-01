from math import pi, sqrt

class Coord:
    # !: remember that the urx uses meters as units.
        # scaling needs to take place when converting between screen and robot coordinates

    # This class is initiated by an x and y- value as well as the x and y size of the robot look area
    # It is the superclass for ScreenCoord and RobotCoord
    # This class is used to convert 2D [x, y] - coordinates into 6D Robot coordinates: [x, y, z, rx, ry, rz]

    def __init__(self, x, y, x_area, y_area):
        self.x = x
        self.y = y
        self.x_area = x_area
        self.y_area = y_area

    def apply_rotation(self, new_coordinates):

        new_coord = new_coordinates
        rotation_factor_x = 0.2
        rotation_factor_y = 0.8
        rotation_angles = [0, 0, 0]
        # print(x_area/2)
        rotation_angles[0] = new_coord[1] / self.y_area / 2 * pi * -rotation_factor_x
        rotation_angles[1] = new_coord[0] / self.x_area / 2 * pi * rotation_factor_y
        # print("rotation angles:", rotation_angles)

        compound_coord = new_coord
        compound_coord.extend(rotation_angles)
        # print("new coordinates with rotation:",new_coord)
        # print("_____________________________")
        #print("new coordinates with rotation:", compound_coord)
        return compound_coord

    def apply_z_value(self, new_coordinates): # takes a coordinate list [x, y, 0] and calculates the corresponding z-value according to the z-factor

        new_coordinates.append(0)  # adds a third coordinate to the list for the Z-value

        z_factor = 0.13  # defines how far the robot goes in Z during the look moves
        xy_offset = sqrt(new_coordinates[0] ** 2 + new_coordinates[1] ** 2)
        # print("xy offset: ", xy_offset)
        overall_dia = sqrt(self.x_area ** 2 + self.y_area ** 2)
        half_dia = overall_dia / 2

        if half_dia == xy_offset:
            new_z = 0
        else:
            new_z = (half_dia) / (half_dia + xy_offset) * z_factor

        new_coordinates[2] = new_z
        # print("new z: ", new_z)

        return new_coordinates


class ScreenCoord(Coord):

    def __init__(self, x, y, x_area, y_area, robot_position, screen_dimensions): #Screencoordinates also require the current robot position
        super().__init__( x, y, x_area, y_area)
        self.xy_pos = robot_position[0: 2]
        self.screen_dimensions = screen_dimensions
        self.size_calibration = 1 / 5000

    def newXY(self):
        # convert the x,y coordinates from the screen coordinate system to the robot coordinate system
        screen_center = [self.screen_dimensions[0] / 2, self.screen_dimensions[1] / 2]
        # calculate position from screen center:
        new_x = (self.x - screen_center[0]) * self.size_calibration
        new_y = (self.y - screen_center[1]) * self.size_calibration

        # add current robot position :
        new_x = new_x + self.xy_pos[0]
        new_y = new_y + self.xy_pos[1]

        new_xy = [new_x, new_y]

        print("new robot position:", new_xy)

        return new_xy


    def convert_screen_coords(self):  # converts screen coordinates to coordinates in the new csys

        new_coord = self.newXY()

        print("new coordinates x+y+z:", new_coord)

        new_coord = self.apply_z_value(new_coord)
        # print("nc: ", new_coord)

        compound_coord = self.apply_rotation(new_coord)

        # print("new coordinates with rotation:", compound_coord)
        # print("_____________________________")

        # TODO: add case for center of c_sys!!
        return compound_coord

    def convert_screen_coords_again(self, coordinates):
        new_coord = coordinates[:2]
        print(new_coord)
        new_coord = self.apply_z_value(new_coord)
        compound_coord = self.apply_rotation(new_coord)

        return compound_coord

class RobotCoord(Coord):

    def convert_robot_coords(self): #converts a list of 2 values [x, y] in the robot csys into full robot coordinates
        # it does this by adding the z value and 3 rotation angles

        new_coord = [self.x, self.y]
        new_coord = self.apply_z_value(new_coord)
        rotation_coords = self.apply_rotation(new_coord)
        # print("rotation_coords", rotation_coords)
        compound_coord = rotation_coords
        # print("new coordinates with rotation:", compound_coord)
        # print("_____________________________")

        return compound_coord
