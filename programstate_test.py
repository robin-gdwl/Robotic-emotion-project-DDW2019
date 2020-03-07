import time
from PROGRAMSTATE import ProgramState

programstate = ProgramState()

while True:
    programstate.level= 0
    time.sleep(5)
    programstate.level= 1
    time.sleep(5)
    programstate.level= 2
    time.sleep(5)
    programstate.level= 3
    time.sleep(5)