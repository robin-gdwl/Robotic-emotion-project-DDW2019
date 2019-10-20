from Robot_Motion import RobotMotion
from String_to_Path import ThingToWrite
import time

Robot = RobotMotion()
Robot.move_paper()
Robot.move_to_write()
origin_offset = [0.05, 0.01]
line_spacing = 0.01

draw_origin = [0,0]
proj_message = ["________________",
                "HOW DO I SEE TECHNOLOGY",
                "WHEN I REALISE ",
                "TECHNOLOGY SEES ME",
                " - ? -",
                "________________",
                "Robotic emotion project",
                "created by:",
                "Robin Godwyll", 
                "and Yang Ni",
                "DDW 2019",
                "www.git.io/JeBir"]



for line in proj_message:
	print("writing: ", line)
	message_coords = ThingToWrite(line).string_to_coordinates(draw_origin,0.0045)  
	Robot.write_results(message_coords)
	draw_origin[1] += line_spacing + 0.002
iteration = 0
Robot.move_paper()
time.sleep(10)
