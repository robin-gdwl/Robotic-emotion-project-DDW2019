from Robot_control import Robot
import time

strings = ["the quick brown fox jumps over the lazy dog __ ", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "ABCabc thijkqr"]
robot = Robot()
robot.initialise_robot()
max_draws = 0

robot.move_home()
#robot.move_to_write()
#robot.write_strings(strings)
i = 0
"""while i < max_draws:
    robot.advance_paper()
    print(i, "of ", max_draws)
    i+=1"""

print("done "*300)

"""

text = "halloooo"

robot.robotUR.textmsg(text)
robot.move_between()
#robot.move_home()
#exit()
robot.move_to_write()
#robot.write_strings(["testing - - -","12345", "6 7 8 9", "! ? _", "abcdefghijklmnopqrstuvw"])
robot.advance_paper()
#robot.move_between()
#robot.wander()
exit()

while True:
    robot.check_position_dist(robot.home_pos)
    print("___"*25)
    robot.check_position_dist(robot.between_pos)
    time.sleep(1)
    print("___" * 25)
    print("___" * 25)
    print("___" * 25)


"""