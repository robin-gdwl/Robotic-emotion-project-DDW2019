# /etc/init.d/sample.py
### BEGIN INIT INFO
# Provides:          sample.py
# Required-Start:    $remote_fs $syslog
# Required-Stop:     $remote_fs $syslog
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: Start daemon at boot time
# Description:       Enable service provided by daemon.
### END INIT INFO

import multiprocessing as mp
import cv2
import time
import sys
import os
import traceback 
from imutils.video import VideoStream
import math

from Face_obj import Face
from Robot_control import Robot
from Videostream import vs
import CONFIG
#from PROGRAMSTATE import ProgramState

# Path to the face-detection model:
#pretrained_model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
#pretrained_model2 = cv2.dnn.readNetFromCaffe("models/RFB-320.prototxt", "models/RFB-320.caffemodel")

#RASPBERRY_BOOL = False
# If this is run on a linux system, it is assumed it runs on a raspberry pi and a picamera will be used.
# If you are using a linux system, with a webcam instead of a raspberry pi delete the following if-statement
def interrupt(channel):
    #global PROGRAMSTATE
    #global ROBOT_ACTION
    global RASPBERRY_BOOL
    global PLAY_PIN
    global RESET_PIN
    global PAUSE_PIN
    print("..."*200)
    print("INTERRUPT!  ROBOTACTION: ", CONFIG.ROBOT_ACTION)
    print("CONFIG.PROGRAMSTATE.level: ", CONFIG.PROGRAMSTATE.level)
    print("CHANNEL: ", channel)
    if channel == PAUSE_PIN:
        pause()
    elif channel == RESET_PIN:
        reset()
    else:
        print("interrupt but unknown button")


if sys.platform == "linux":
    #global PROGRAMSTATE
    CONFIG.RASPBERRY_BOOL = True
    import picamera
    from picamera.array import PiRGBArray
    import RPi.GPIO as GPIO
    
    # Button Pins 
    PLAY_PIN = 11
    PAUSE_PIN = 13
    RESET_PIN = 15
    
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PAUSE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    #GPIO.setup(RESET_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(PLAY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    
    GPIO.add_event_detect(PAUSE_PIN, GPIO.RISING, callback=interrupt, bouncetime = 1000)
    #GPIO.add_event_detect(RESET_PIN, GPIO.RISING, callback=interrupt, bouncetime = 1000)

def check_exhibit_time():
    pass

def pause():
    #global PROGRAMSTATE
    #global ROBOT_ACTION
    #global RASPBERRY_BOOL
    global PLAY_PIN
    global RESET_PIN 
    global robot
    
    CONFIG.PROGRAMSTATE.level = 3
    print("pause")
    #robot.robotUR.waitRobotIdleOrStopFlag()
    robot.stop_safe()
    #robot.robotUR.stopj(robot.accel)
    time.sleep(1)
    #robot.move_safe(CONFIG.ROBOT_ACTION)
    robot.move_home()

    CONFIG.PROGRAMSTATE.level = 1
    
    if CONFIG.RASPBERRY_BOOL:
    #if False:
        print("waiting to continue")
        print("----" * 5)
        GPIO.wait_for_edge(PLAY_PIN, GPIO.BOTH)
        CONFIG.PROGRAMSTATE.level = 0
        
        
    else:  # what to do if this runs on a mac and there is no button 
        print("waiting 10s to continue")
        print("----"*5)
        time.sleep(10)
        CONFIG.PROGRAMSTATE.level = 0

    robot.robotUR.reset_error()
    print("continuing")
    robot.start_rtde()
    time.sleep(1)

def reset(waitforplay=False, reboot=True):
    # do reset procedure
    #create reboot interrupt
    #global PROGRAMSTATE
    CONFIG.PROGRAMSTATE.level = 4
    robot.robotUR.stopj(robot.accel / 5)
    time.sleep(1)
    robot.move_safe(CONFIG.ROBOT_ACTION)
    robot.move_home()
    #if reboot: reboot
    if reboot:
        os.system('sudo reboot')
    
    global RASPBERRY_BOOL
    global PLAY_PIN
    global RESET_PIN
    
    if waitforplay:
        GPIO.wait_for_edge(PLAY_PIN, GPIO.BOTH)
    pass

def wait4play():
    CONFIG.PROGRAMSTATE.level = 1

    if CONFIG.RASPBERRY_BOOL:
        print("waiting to continue")
        print(CONFIG.PROGRAMSTATE.level)
        print("----" * 5)
        GPIO.wait_for_edge(PLAY_PIN, GPIO.BOTH)
        CONFIG.PROGRAMSTATE.level = 3
        print("Play Button Pressed")
        print("----" * 5)

# ___________________________________________________________________________________________________________________________________

#starting Program: 
# 1. : Set programstate to Starting
CONFIG.PROGRAMSTATE.level = -1
print("___"*20)
print("Starting Process")
time.sleep(1)
wait4play()

print("initialising loop")
#PROGRAMSTATE = ProgramState()
CONFIG.PROGRAMSTATE.level = 3
robot = Robot()
robot.initialise_robot()
#robot.move_between()
robot.move_home()

robot.current_row = 0
#robot.start_rtde()
time.sleep(1)

print("____"*50)


def loop():
    #global PROGRAMSTATE
    print("looop")
    CONFIG.PROGRAMSTATE.level = 0
    try: 
        while True :
            print("CONFIG.PROGRAMSTATE.level at top of loop: ", CONFIG.PROGRAMSTATE.level)
            if CONFIG.PROGRAMSTATE.level == 0:
                
                robot.wander()
                cln_img, face_img, face_box, face_pos  = robot.follow_face(close=False)
                
                if not face_pos:
                    continue
                else:
                    filename = str(time.time()) + ".png"
                    cv2.imwrite(filename, face_img)
                    filename = "cln-" + str(time.time()) + ".png"
                    cv2.imwrite(filename, cln_img)
                print("face follow done")
            
            if CONFIG.PROGRAMSTATE.level == 0:
                time.sleep(0.1)
                landmark_queue = mp.Queue()
                emotion_queue = mp.Queue()
                robot.robotUR.textmsg("face found, moving to write")
                robot.move_to_write(robot.current_row)
                if robot.check_paper():
                    robot.move_to_write(robot.current_row)
                robot.create_coordinates(cln_img, face_box)
                
            if CONFIG.PROGRAMSTATE.level == 0:
                robot.check_paper()
            
                robot.move_home()
            
            if CONFIG.PROGRAMSTATE.level == 0:
                robot.start_rtde()
            else:
                time.sleep(1)
                continue
            
    except Exception as e:
        print("ERROR: ", e)
        traceback.print_exc()
        print("closing robot connection")
        robot.robotUR.close()
        

def main():
    loop()
        
if __name__ == '__main__':
    main()
    