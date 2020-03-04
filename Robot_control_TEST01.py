from Robot_control import Robot


strings = ["the quick brown fox jumps over the lazy dog __ ", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "ABCabc thijkqr"]
robot = Robot()
robot.initialise_robot()

robot.move_to_write()
#robot.write_strings(strings)
robot.advance_paper()
robot.advance_paper()
robot.advance_paper()
robot.advance_paper()
robot.advance_paper()
robot.advance_paper()
robot.advance_paper()
robot.advance_paper()



