from math import pi, sqrt

class Coord:

    def __init__(self, x, y, x_area, y_area):
        self.x = x
        self.y = y
        self.x_area = x_area
        self.y_area = y_area

    def newXY(self):
        # convert the x,y coordinates from the screen coordinate system to the robot coordinate system
        # the screen coordinate system starts at the top left corner,
        # the robot coordinate system starts from the center
        # these two formulas simply convert from one to the other.
        new_x = self.x - (1 / 2 * self.x_area)
        new_y = self.y - (1 / 2 * self.y_area)
        new_XY = [new_x, new_y]

        return new_XY

    def convert_coord(self):  # converts screen coordinates to coordinates in the new csys

        new_coord = self.newXY()
        new_coord.append(0) # adds a third coordinate to the list for the Z-value

        print("new coordinates x+y:", new_coord)

        z_factor = 0.2  # defines how far the robot goes in Z
        xy_offset = sqrt(new_coord[0] **2 + new_coord[1] **2)
        # print("xy offset: ", xy_offset)
        overall_dia = sqrt(self.x_area **2 + self.y_area **2)
        half_dia = overall_dia / 2


        if half_dia == xy_offset:
            new_z = 0
        else:
            new_z = (half_dia) / (half_dia + xy_offset ) * z_factor

        new_coord[2] = new_z
        print("new z: ", new_z)

        rotation_factor = 0.1
        rotation_angles = [0, 0, 0]
        # print(x_area/2)
        rotation_angles[0] = new_coord[1] / self.y_area/2 * pi * -rotation_factor
        rotation_angles[1] = new_coord[0] / self.x_area/2 * pi * rotation_factor
        print("rotation angles:", rotation_angles)

        compound_coord = new_coord
        compound_coord.extend(rotation_angles)
        # print("new coordinates with rotation:",compound_coord)
        print("_____________________________")

        # TODO: add case for center of c_sys!!
        return compound_coord