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
HOURS_TO_TRAIN = 3.5
MAX_MEM = 750
BATCH_SIZE = 32 # 16 on my laptop due to lesser GPU
CONTINUE_TRAINING = True
IN_SITU_TRAINING = True
IN_SITU_BATCH_SIZE = 4
DESIRED_FPS = 4
STR_FPS = str(DESIRED_FPS).replace(".", "_")

if __name__ == "__main__":
    
    # get screenshot size
    sample_screenshot = get_screenshot(SHRINK_FACTOR)
    screen_shape_shrunk = sample_screenshot.shape
    print("is cuda availible?: " + str(torch.cuda.is_available()))
    # uncomment for visual inspection of a sample screenshot
    # manually_inspect_screenshot_sample(shrink_factor)
    

    # Make an agent that will be playing and learning
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=screen_shape_shrunk, batch_size=BATCH_SIZE,
                    n_actions=NUM_ACTIONS, max_mem_size=MAX_MEM, eps_end=0.005, eps_dec=1e-3, need_CNN=True)
    if (CONTINUE_TRAINING):
        total_train_hours = 1
        initial_hours = total_train_hours
        total_train_hours_str = str(total_train_hours).replace(".", "_")
        agent.load_models("deepQLastContinuous" + total_train_hours_str + "Hours" + STR_FPS + "FPS.pt", "targetQLastContinuous" + total_train_hours_str + "Hours" + STR_FPS + "FPS.pt")

    # how long to train, and a counter to tell us how many games that was. Traditionally, we
    # train for a number of games, not an amount of real time, but this program is going to
    # be the ONLY thing youre computer will be doing at a time, so I figure the user wants
    # to know how long of a lunch break to take
    seconds_to_train = HOURS_TO_TRAIN*60*60 + 5
    num_games_completed = 0

    # some things for recording data
    scores, eps_history = [], []
    max_time_survived = 0
    learn_cycles_snuck_in = 0
    memories_formed = 0

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
    
    while (time() - global_time < seconds_to_train):
        
        # reset for a new game and get our first observation
        game_start_time = time()
        fps_time = time()
        learn_cycles_snuck_in = 0
        score = 0
        dead = False
        observation = get_screenshot(SHRINK_FACTOR)
        while (not dead):
            
            # check time, then wait to ensure constant input rate
            time_since_last_input = time() - fps_time
            max_potiential_fps = (1 / time_since_last_input)
            print("max potiential FPS: " + str(max_potiential_fps))
            if (time_since_last_input < time_between_frames):
                sleep(time_between_frames - time_since_last_input)

            # pick an action and do it
            action = agent.choose_action(observation)
            perform_maneuver(action)
            fps_time = time()

            # evaluate our move
            new_observation = get_screenshot(SHRINK_FACTOR)
            dead = are_you_dead_mon(new_observation)
            reward = 0 if dead else time() - game_start_time
            score += reward


            # store what happened. Only learn if your PC can handle it
            agent.store_transition(observation, action, reward, new_observation, dead)
            memories_formed += 1
            if (IN_SITU_TRAINING and memories_formed > BATCH_SIZE):
                agent.batch_size = IN_SITU_BATCH_SIZE
                agent.learn()
                agent.batch_size = BATCH_SIZE
            observation = new_observation

        # record data about that game
        death_time = time()
        time_survived = time() - game_start_time
        if (time_survived > 75):
            minutes_took = int((time() - global_time) /60)
            agent.save_models("deepQPASSED" + str(minutes_took) + "Minutes" + str(num_games_completed) + "Games.pt",
                                "targetQPASSED" + str(minutes_took) + "Minutes" + str(num_games_completed) + "Games.pt")
        max_time_survived = time_survived if time_survived > max_time_survived else max_time_survived
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        num_games_completed += 1
        
        # When the game ends, we have about 1.25 seconds before Super Hexagon
        # allows us to play again. Lets use this time to learn
        while (time() - death_time < 1.25):
            agent.learn()
            learn_cycles_snuck_in += 1

        # save models every hour, in case the power goes out or something
        if CONTINUE_TRAINING and (((time()-global_time)/60/60) + initial_hours) - total_train_hours >= 1.0:
            total_train_hours += 1
            total_train_hours_str = str(total_train_hours).replace(".", "_")
            agent.save_models("deepQLastContinuous" + total_train_hours_str + "Hours" + STR_FPS + "FPS.pt", "targetQLastContinuous" + total_train_hours_str + "Hours" + STR_FPS + "FPS.pt")

        # status update to console, manage memory, and start next game
        end_of_game_status_report(learn_cycles_snuck_in, num_games_completed, score, avg_score, agent, max_time_survived, global_time)
        garbage_collect(agent)
        pydirectinput.press('space')

    # save our model so we can pick up where we left off
    if (CONTINUE_TRAINING):
        total_train_hours_str = str(total_train_hours).replace(".", "_")
        agent.save_models("deepQLastContinuous" + total_train_hours_str + "Hours" + STR_FPS + "FPS.pt", "targetQLastContinuous" + total_train_hours_str + "Hours" + STR_FPS + "FPS.pt")
    elif (HOURS_TO_TRAIN >= 1):
        time_string = "%.2f" % HOURS_TO_TRAIN
        time_string = time_string.replace('.', '_')
        agent.save_models("deepQLast" + time_string + "Hours" + STR_FPS + "FPS.pt", "targetQLast" + time_string + "Hours" + STR_FPS + "FPS.pt")
    else: 
        minutes_trained = int(HOURS_TO_TRAIN * 60)
        agent.save_models("deepQLast" + str(minutes_trained) + "Minutes" + STR_FPS + "FPS.pt", "targetQLast" + str(minutes_trained) + "Minutes" + STR_FPS + "FPS.pt")

    # Graph x, the games, against the scores and the eps_history
    x =[i+1 for i in range(num_games_completed)]
    graph_results(x, scores, eps_history, num_games_completed, HOURS_TO_TRAIN, STR_FPS)

    print("models and data saved, done.")