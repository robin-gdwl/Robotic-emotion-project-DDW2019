from imutils.video import VideoStream
import CONFIG
import sys


vs = VideoStream(src= 0 ,
                 usePiCamera= CONFIG.RASPBERRY_BOOL,
                 resolution=CONFIG.VIDEO_RESOLUTION,
                 framerate = 12,
                 meter_mode = "auto",
                 exposure_mode ="auto",
                 shutter_speed = 5900,
                 exposure_compensation = 3,
                 rotation = 180).start()