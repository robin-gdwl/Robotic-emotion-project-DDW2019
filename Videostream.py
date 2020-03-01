from imutils.video import VideoStream
import CONFIG
import sys


vs = VideoStream(src= 0 ,
                 usePiCamera= CONFIG.RASPBERRY_BOOL,
                 resolution=CONFIG.VIDEO_RESOLUTION,
                 framerate = 13,
                 meter_mode = "backlit",
                 exposure_mode ="auto",
                 shutter_speed = 8900,
                 exposure_compensation = 2,
                 rotation = 180).start()