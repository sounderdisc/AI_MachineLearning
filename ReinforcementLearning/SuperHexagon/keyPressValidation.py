# This file will give the user 5 seconds before pressing a sequence
# of keyboard inputs until it does for 10 games. Good to make sure
# the input and game end detection is working before running the
# more complex and resource intense full agent 
import torch
import numpy as np
import pydirectinput
from HelperFunctions import *

if __name__ == "__main__":
    
    print("is cuda availible?: " + str(torch.cuda.is_available()))

    screenshot = get_screenshot(1.0, grayscale=False)
    screen_shape = screenshot.shape
    print(screen_shape)
    dead = False
    current_action = 0

    play_count_down()
    pydirectinput.press('space')
    perform_maneuver(26)
    perform_maneuver(27)
    for i in range(10):
        
        print("sequence " + str(i+1))

        while (not dead):
            perform_maneuver(current_action % NUM_ACTIONS)
            current_action += 1
            
            screenshot = get_screenshot(1.0, index=i)
            dead = are_you_dead_mon(screenshot)

        dead = False
        sleep(1.25)
        pydirectinput.press('space')

    
    cv.destroyAllWindows()

    # pydirectinput.press('left')
    # sleep(0.025)
    # pydirectinput.press('right')
    # sleep(0.025)