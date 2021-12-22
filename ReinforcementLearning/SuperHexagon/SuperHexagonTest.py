import torch
import numpy as np
from time import time
from time import sleep
import pydirectinput
import sys ; sys.path.append('..')
from Agent import Agent
from HelperFunctions import *


# Constrants to control memory use and training
# Your system's vRAM on your graphics card will determine your
# batch size, and your system RAM will determine your maximum
# replay memory size
SHRINK_FACTOR = 1.0
GAMES_TO_TEST = 10
MAX_MEM = 750
BATCH_SIZE = 32
IN_SITU_TRAINING = True
IN_SITU_BATCH_SIZE = 4
DESIRED_FPS = 4
STR_FPS = str(DESIRED_FPS).replace(".", "_")
SOLVED_TIME = 75

if __name__ == "__main__":
    
    # get screenshot size
    sample_screenshot = get_screenshot(SHRINK_FACTOR)
    screen_shape_shrunk = sample_screenshot.shape
    print("is cuda availible?: " + str(torch.cuda.is_available()))
    # uncomment for visual inspection of a sample screenshot
    # manually_inspect_screenshot_sample(shrink_factor)
    

    # Make an agent and choose which model to load
    agent = Agent(gamma=0.99, epsilon=0.0, lr=0.0001, input_dims=screen_shape_shrunk, batch_size=BATCH_SIZE,
                    n_actions=NUM_ACTIONS, max_mem_size=MAX_MEM, eps_end=0.0, eps_dec=1e-3, need_CNN=True)
    time_trained_number = "3_50" # "1_5"
    time_trained_units = "Hours" # "Minutes"
    agent.load_models("deepQLast" + time_trained_number + time_trained_units + STR_FPS + "FPS.pt", "TargetQLast" + time_trained_number + time_trained_units + STR_FPS + "FPS.pt")

    # some things for recording data
    scores, eps_history = [], []
    max_time_survived = 0

    # Time will be used as a score, and to ensure a constant amount of time between actions
    time_between_frames = 1 / DESIRED_FPS
    fps_time = time()
    game_start_time = time()
    death_time = time()
    global_time = time()
    
    # We are waiting here to give the user time to tab back into Super Hexagon
    # before the program begins playing. ImageGrab.grab() only takes whats on
    # the frontmost window. pydirectinput only inputs intot he frontmost window
    # The program will press space at the end of the timer
    play_count_down()
    pydirectinput.press('space')
    
    for i in range(GAMES_TO_TEST):
        
        # reset for a new game and get our first observation
        game_start_time = time()
        fps_time = time()
        score = 0
        dead = False
        observation = get_screenshot(SHRINK_FACTOR)
        while (not dead):
            
            # check time, then wait to ensure constant input rate
            time_since_last_input = time() - fps_time
            if (time_since_last_input < time_between_frames):
                sleep(time_between_frames - time_since_last_input)

            # pick an action and do it
            action = agent.choose_action(observation)
            perform_maneuver(action)
            fps_time = time()

            # observe new game state
            new_observation = get_screenshot(SHRINK_FACTOR)
            dead = are_you_dead_mon(new_observation)
            observation = new_observation

            # Early termination, since Super Hexagon is an infinite game
            time_survived = time() - game_start_time
            if (time_survived >= SOLVED_TIME):
                pydirectinput.press('esc')
                break

        # record data about that game
        death_time = time()
        time_survived = time() - game_start_time
        max_time_survived = time_survived if time_survived > max_time_survived else max_time_survived
        scores.append(time_survived)
        eps_history.append(agent.epsilon)
        avg_time_survived = np.mean(scores[-100:])
        
        # When the game ends, we have about 1.25 seconds before Super Hexagon
        # allows us to play again.
        sleep(1.25)

        # status update to console, manage memory, and start next game
        end_of_game_status_report_test(i, time_survived, avg_time_survived, agent.epsilon, max_time_survived)
        garbage_collect(agent)
        pydirectinput.press('space')

    # Graph x, the games, against the scores and the eps_history
    x =[i+1 for i in range(GAMES_TO_TEST)]
    graph_results_test(x, scores, eps_history, time_trained_number, time_trained_units, STR_FPS)

    print("models and data saved, done.")