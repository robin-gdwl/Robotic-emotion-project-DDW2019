import time
from PROGRAMSTATE import ProgramState

programstate = ProgramState()

while True:
    print("loop new ")
    print(programstate.level)
    programstate.level= 0
    print(programstate.level)
    time.sleep(5)
    
    programstate.level= 1
    print(programstate.level)
    time.sleep(5)
    programstate.level= 2
    print(programstate.level)
    time.sleep(5)
    programstate.level= 3
    print(programstate.level)
    time.sleep(5)