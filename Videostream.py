from imutils.video import VideoStream
import CONFIG
import sys

if sys.platform == "linux":
    RASPBERRY_BOOL = True
    import picamera
    from picamera.array import PiRGBArray


vs = VideoStream(src= 0 ,
                 usePiCamera= RASPBERRY_BOOL,
                 resolution=CONFIG.VIDEO_RESOLUTION,
                 framerate = 13,
                 meter_mode = "backlit",
                 exposure_mode ="auto",
                 shutter_speed = 8900,
                 exposure_compensation = 2,
                 rotation = 0).start()