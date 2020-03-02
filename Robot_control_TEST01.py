from Robot_control import Robot


strings = ["Test 12345", "00000--??????", "the quick brown fox jumps over the lazy dog __ "]
robot = Robot()
robot.initialise_robot()

robot.move_to_write()
robot.write_strings(strings)
robot.advance_paper()


