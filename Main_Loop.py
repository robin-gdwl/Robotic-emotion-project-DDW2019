import multiprocessing as mp
import cv2
import time
import sys
import traceback 
from imutils.video import VideoStream
import math

from Face_obj import Face
from Robot_control import Robot
import CONFIG

# Path to the face-detection model:
pretrained_model = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
pretrained_model2 = cv2.dnn.readNetFromCaffe("models/RFB-320.prototxt", "models/RFB-320.caffemodel")



RASPBERRY_BOOL = False
# If this is run on a linux system, it is assumed it runs on a raspberry pi and a picamera will be used.
# If you are using a linux system, with a webcam instead of a raspberry pi delete the following if-statement
def interrupt(channel):
    global PROGRAMSTATE
    global ROBOT_ACTION
    global RASPBERRY_BOOL
    global PLAY_PIN
    global RESET_PIN
    global PAUSE_PIN
    
    print("INTERRUPT!  ROBOTACTION: ", ROBOT_ACTION)
    print("PROGRAMSTATE: ", PROGRAMSTATE)
    if channel == PAUSE_PIN:
        pause()
    elif channel == RESET_PIN:
        reset()
    else:
        print("interrupt but unknown button")


if sys.platform == "linux":
    RASPBERRY_BOOL = True
    import picamera
    from picamera.array import PiRGBArray
    import RPi.GPIO as GPIO

    PAUSE_PIN = 8
    RESET_PIN = 10
    PLAY_PIN = 12
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(PAUSE_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(RESET_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(PLAY_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.add_event_detect(PAUSE_PIN, GPIO.RISING, callback=interrupt, bouncetime = 500)
    GPIO.add_event_detect(RESET_PIN, GPIO.RISING, callback=interrupt, bouncetime = 500)

vs = VideoStream(src= 0 ,
                 usePiCamera= RASPBERRY_BOOL,
                 resolution=CONFIG.VIDEO_RESOLUTION,
                 framerate = 13,
                 meter_mode = "backlit",
                 exposure_mode ="auto",
                 shutter_speed = 8900,
                 exposure_compensation = 2,
                 rotation = 0).start()

# PROGRAMSTATE: 0 = running , 1 = pause, 2 = error
CONFIG.PROGRAMSTATE = 0
CONFIG.ROBOT_ACTION = 0

def check_exhibit_time():
    pass

        
def pause():
    global PROGRAMSTATE
    global ROBOT_ACTION
    global RASPBERRY_BOOL
    global PLAY_PIN
    global RESET_PIN 
    
    PROGRAMSTATE = 1
    print("pause")
    robot.robotUR.stopj(robot.accel/5)
    time.sleep(1)
    robot.move_safe(ROBOT_ACTION)
    robot.move_home()
    
    if RASPBERRY_BOOL:
    #if False:
        print("waiting to continue")
        print("----" * 5)
        GPIO.wait_for_edge(PLAY_PIN, GPIO.BOTH)
        PROGRAMSTATE = 0
        
        print("continuing")
        robot.start_rtde()
        time.sleep(1)
        
    else:  # what to do if this runs on a mac and there is no button 
        print("waiting 10s to continue")
        print("----"*5)
        time.sleep(10)
        PROGRAMSTATE = 0

        print("continuing")
        robot.start_rtde()
        time.sleep(1)

def reset():
    pass

#robot_ip = "10.211.55.5"
robot_ip = "192.168.178.20"

robot = Robot(robot_ip)
robot.initialise_robot()
robot.move_home()

robot.current_row = 0
robot.start_rtde()
time.sleep(1)

print("____"*800)

def main():    
    PROGRAMSTATE = 0
    try: 
        while True :
            
            if PROGRAMSTATE == 0:
                
                robot.wander()
                face_img, face_box, face_pos  = robot.follow_face(close=False)
                #cv2.imwrite("testface.png", face_img)
                
                
                if not face_pos:
                    continue
                print("face follow done")
                #time.sleep(0.1)
                landmark_queue = mp.Queue()
                emotion_queue = mp.Queue()
            
                robot.move_to_write(robot.current_row)
                robot.create_coordinates(face_img, face_box)
                
                robot.current_row += 1
                robot.check_paper()
            
                robot.move_home()
                robot.start_rtde()
            else:
                time.sleep(1)
                continue
            
    except Exception as e:
        print("ERROR: ", e)
        traceback.print_exc()
        print("closing robot connection")
        robot.robotUR.close()
        
if __name__ == '__main__':
    main()
    