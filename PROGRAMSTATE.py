import threading
import math
import sys
import time 
import PI_PINS
from itertools import compress

# WHY DOES THIS NOT SYNC with git ???? ?????

# if CONFIG.RASPBERRY_BOOL:
Pi_bool = False
if sys.platform == "linux":
    Pi_bool = True
    import RPi.GPIO as GPIO
    

class ProgramState:
    
    def __init__(self,initial_lvl = 0 ):
        self.__level = 0
    
        self.led_values_dict = {-1: "blue flash fast",
                                0: "green",
                                1: "green flash",
                                2: "Red flash",
                                3: "blue flash",
                                4: "red"}
        self.led_pin_dict = {"red": PI_PINS.RED_PIN,
                             "green": PI_PINS.GREEN_PIN,
                             "blue": PI_PINS.BLUE_PIN}
        self.led_pin_list = [self.led_pin_dict["red"],
                        self.led_pin_dict["green"],
                        self.led_pin_dict["blue"]]
        
        self.activate_led = Pi_bool
        if self.activate_led:
            self.__setup_led()
            self._led_thread()

        self.level_dict = {-1: "Starting",
                           0: "Playing",
                           1: "Paused and ready",
                           2: "Error",
                           3: "Performing action, wait for next colour",
                           4: "Resetting"}
        self.level = initial_lvl
        # self.action = 0 
    
    def __setup_led(self):
        if self.activate_led == True:  # and CONFIG.RASPBERRY_BOOL:
            GPIO.setmode(GPIO.BOARD) 
            #GPIO.cleanup()
            GPIO.setup(PI_PINS.RED_PIN, GPIO.OUT)
            GPIO.output(PI_PINS.RED_PIN, 0)

            GPIO.setup(PI_PINS.GREEN_PIN, GPIO.OUT)
            GPIO.output(PI_PINS.GREEN_PIN, 0)

            GPIO.setup(PI_PINS.BLUE_PIN, GPIO.OUT)
            GPIO.output(PI_PINS.BLUE_PIN, 0)
        else:
            self.activate_led = False

    def print_lvl(self):
        level_p = self.level_dict[self.level]
        print(level_p)
        return level_p
    
    def _led_thread(self):
        led_t = threading.Thread(target= self.__set_led)
        led_t.start()
    
    def __set_led(self):
        if self.activate_led:
            print("LED thread started")
            previous_level = self.level
            while self.activate_led:             
                # print(self.led_values_dict[self.level])
                if self.__level == 0:
                    """if self.level == previous_level:
                        # no level change do not change led
                        pass"""
                    # make LED green 
                    self.__led_change(Green=1)
                elif self.__level == -1:
                    self.__led_change(Blue=1,flash=True, interval=0.1)   
                    # flash LED Blue very fast on startup 
                elif self.__level == 1:
                    # flash LED green
                    self.__led_change(Green=1, flash = True)
                elif self.__level == 2:
                    # flash LED red
                    self.__led_change(Red=1, flash=True)
                elif self.__level == 3:
                    # flash LED blue
                    self.__led_change(Blue=1, flash=True)
                elif self.__level == 4:
                    # make LED red
                    self.__led_change(Red=1)
                else:
                    print("unknown level!, no led change")
                    # something to do if an unknown level
    
    def __led_change(self, Red=0, Green=0, Blue=0, flash = False, interval=0.3):
        led_bools = [Red,Green,Blue]
        #print("led_bools: ", led_bools)
        
        GPIO.output(self.led_pin_list, led_bools)
        time.sleep(interval)
        if flash:
            GPIO.output(self.led_pin_list, 0)
            time.sleep(interval)
            
            
    @property 
    def level(self):
        return self.__level
        
    @level.setter
    def level(self, new_lvl):
        if new_lvl in self.level_dict:
            
            self.__level = new_lvl
        else: 
            print("unknown level, no level set")
        
        
        
    