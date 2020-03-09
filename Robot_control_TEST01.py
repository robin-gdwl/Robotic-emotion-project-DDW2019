from Robot_control import Robot


strings = ["the quick brown fox jumps over the lazy dog __ ", "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "ABCabc thijkqr"]
robot = Robot()
robot.initialise_robot()
max_draws = 200


#robot.move_to_write()
#robot.write_strings(strings)
i = 0
"""while i <= max_draws:
    robot.advance_paper()
    print(i, "of ", max_draws)
    i+=1

print("done "*300)"""

text = "halloooo"

robot.robotUR.textmsg(text)



