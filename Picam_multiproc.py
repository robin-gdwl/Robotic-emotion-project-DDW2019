from imutils.video import VideoStream
import picamera
from picamera.array import PiRGBArray
import multiprocessing as mp
import time
import cv2

vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
while True:
    # read the frame from the camera and send it to the server
    timer = time.time()
    frame = vs.read()
    cv2.imshow("Frame", frame)
    print("image processed in", time.time() - timer)
    cv2.waitKey(1)  # this defines how long each frame is shown




class CameraCapture:

    def __init__(self):
        self.camera = picamera.PiCamera()
        self.camera.resolution = (800, 800)
        self.rawCapture = PiRGBArray(self.camera)
        self.frame = []

    def takeimg(self):

        while True:
            timer = time.time()
            self.camera.shutter_speed = 10
            self.camera.capture(self.rawCapture, format="bgr")  # capture the image
            print("image taken in", time.time() - timer)
            image = self.rawCapture.array
            print("image processed in", time.time()-timer)

            self.frame = image
            self.rawCapture.truncate(0)
            print("exposure time: ", self.camera.exposure_speed)
            #cv2.imshow("Frame", image)
            #cv2.waitKey(1)  # this defines how long each frame is shown

def makeimg():
    while True:
        print(cam.frame)
        time.sleep(2)

'''
cam = CameraCapture()

p1 = mp.Process(target=cam.takeimg())
p2 = mp.Process(target=makeimg)
p1.start()
p2.start()

'''






