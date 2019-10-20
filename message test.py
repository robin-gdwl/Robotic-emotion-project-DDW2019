from String_to_Path import ThingToWrite

draw_origin = [0,0]
line_spacing = 0.01

proj_message = ["___________________",
                "HOW DO I SEE TECHNOLOGY",
                "WHEN I REALISE ",
                "TECHNOLOGY SEES ME",
                " - ? -",
                "______________________",
                "Robotic emotion project",
                "created by:",
                "Robin Godwyll and Yang Ni",
                "Dutch Design Week 2019",
                "www.git.io/JeBir"]  # This message is written by the robot after a certain amount of evaluations



for line in proj_message:
    print("writing: ", line)
    message_coords = ThingToWrite(line).string_to_coordinates(draw_origin)  # add origin here
    #Robot.write_results(message_coords)
    draw_origin[1] += line_spacing + 0.002


#message_coords = ThingToWrite(proj_message).string_to_coordinates(draw_origin)

print(proj_message)
