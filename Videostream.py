from imutils.video import VideoStream
import CONFIG
import sys


vs = VideoStream(src= 0 ,
                 usePiCamera= CONFIG.RASPBERRY_BOOL,
                 resolution=CONFIG.VIDEO_RESOLUTION,
                 framerate = 15,
                 meter_mode = "backlit",
                 exposure_mode ="auto",
                 shutter_speed = 10000,
                 exposure_compensation = 3,
                 iso = 1600,
                 rotation = 180).start()