import cv2 as cv
import numpy as np
import gc
import pydirectinput
from time import time
from time import sleep
from PIL import ImageGrab
from PIL import Image
import matplotlib.pyplot as plt


NUM_ACTIONS = 29


def get_screenshot(shrink_factor, debug=False, index=-1, grayscale=False):
    # screenshot = ImageGrab.grab()
    screenshot = ImageGrab.grab(bbox=(0, 24, 764, 500)) # bbox=(left_x, top_y, right_x, bottom_y)
    screenshot = np.array(screenshot)
    if (1 - shrink_factor > 0.0001):
        screenshot_shrunk = cv.resize(screenshot, (0, 0), fx=shrink_factor, fy=shrink_factor)
    else:
        screenshot_shrunk = screenshot
    # greyscale will reduce channels from 3 to 1 and help save memory
    if (grayscale):
        screenshot_shrunk = cv.cvtColor(screenshot_shrunk, cv.COLOR_BGR2GRAY)
        screenshot_shrunk = np.expand_dims(screenshot_shrunk, axis=2)
    # saving to disk will tank performance, since read/write is slow. Only here for debugging
    if (debug):
        im_normal = Image.fromarray(screenshot)
        im_normal.save("fullSizedCapture.png")
        im_shrunk = Image.fromarray(screenshot_shrunk)
        im_shrunk.save("shrunkSizedCapture"+str(index)+".png")
        # cv.imshow("debug Shrunk", screenshot_shrunk)
    # The shape by default is (height, width, channels) but pytorch will want it
    # to be of shape (channels, height, width) later, so we'll change it here
    screenshot_shrunk_transposed = np.transpose(screenshot_shrunk, axes=[2,0,1])
    return screenshot_shrunk_transposed

# ever watch Cool Runnings?
# https://www.youtube.com/watch?v=kEIfp1Uvoqc&ab_channel=CallumSmith
def are_you_dead_mon(screenshot, shrink_factor=1, debug=False):
    # In Super Hexagon, there is a bit of UI area that is black if the
    # game is running, and a non-black color between games.
    w = int(3 * shrink_factor)
    h = int(3 * shrink_factor)
    # w=60
    # h=20
    pixel_loc = [h, w]
    if (screenshot.shape[0] == 3):
        pixel_val = [screenshot[0][h][w], screenshot[1][h][w], screenshot[2][h][w]]
        pixel_sum = np.sum(pixel_val)
    else:
        pixel_val = [screenshot[0][h][w]]
        pixel_sum = screenshot[0][h][w]
    if (debug):
        print("pixel_loc" + str(pixel_loc) + "\npixel_val: " + str(pixel_val) +"\npixel_sum: " + str(pixel_sum))
    if (pixel_sum > 25):
        if (debug):
            print("DEAD")
        return True
    else:
        if (debug):
            print("alive")
        return False

def perform_maneuver(action):
    # press for 1 milisecond
    if (action == 0):
        pydirectinput.keyDown('left')
        sleep(0.001)
        pydirectinput.keyUp('left')
    elif (action == 1):
        pydirectinput.keyDown('right')
        sleep(0.001)
        pydirectinput.keyUp('right')
    # press for 5 milisecond
    elif (action == 2):
        pydirectinput.keyDown('left')
        sleep(0.005)
        pydirectinput.keyUp('left')
    elif (action == 3):
        pydirectinput.keyDown('right')
        sleep(0.005)
        pydirectinput.keyUp('right')
    # press for 10 milisecond
    elif (action == 4):
        pydirectinput.keyDown('left')
        sleep(0.01)
        pydirectinput.keyUp('left')
    elif (action == 5):
        pydirectinput.keyDown('right')
        sleep(0.01)
        pydirectinput.keyUp('right')
    # press for 50 milisecond
    elif (action == 6):
        pydirectinput.keyDown('left')
        sleep(0.05)
        pydirectinput.keyUp('left')
    elif (action == 7):
        pydirectinput.keyDown('right')
        sleep(0.05)
        pydirectinput.keyUp('right')
    # press for 100 milisecond
    elif (action == 8):
        pydirectinput.keyDown('left')
        sleep(0.1)
        pydirectinput.keyUp('left')
    elif (action == 9):
        pydirectinput.keyDown('right')
        sleep(0.1)
        pydirectinput.keyUp('right')
    # press for 150 miliseconds
    elif (action == 10):
        pydirectinput.keyDown('left')
        sleep(0.15)
        pydirectinput.keyUp('left')
    elif (action == 11):
        pydirectinput.keyDown('right')
        sleep(0.15)
        pydirectinput.keyUp('right')
    # press for 200 miliseconds
    elif (action == 12):
        pydirectinput.keyDown('left')
        sleep(0.2)
        pydirectinput.keyUp('left')
    elif (action == 13):
        pydirectinput.keyDown('right')
        sleep(0.2)
        pydirectinput.keyUp('right')
    # press for 250 miliseconds
    elif (action == 14):
        pydirectinput.keyDown('left')
        sleep(0.25)
        pydirectinput.keyUp('left')
    elif (action == 15):
        pydirectinput.keyDown('right')
        sleep(0.25)
        pydirectinput.keyUp('right')
    # press for 300 milliseconds
    elif (action == 16):
        pydirectinput.keyDown('left')
        sleep(0.3)
        pydirectinput.keyUp('left')
    elif (action == 17):
        pydirectinput.keyDown('right')
        sleep(0.3)
        pydirectinput.keyUp('right')
    # press for 350 milliseconds
    elif (action == 18):
        pydirectinput.keyDown('left')
        sleep(0.35)
        pydirectinput.keyUp('left')
    elif (action == 19):
        pydirectinput.keyDown('right')
        sleep(0.35)
        pydirectinput.keyUp('right')
    # press for 400 milliseconds
    elif (action == 20):
        pydirectinput.keyDown('left')
        sleep(0.4)
        pydirectinput.keyUp('left')
    elif (action == 21):
        pydirectinput.keyDown('right')
        sleep(0.4)
        pydirectinput.keyUp('right')
    # press for 450 milliseconds
    elif (action == 22):
        pydirectinput.keyDown('left')
        sleep(0.45)
        pydirectinput.keyUp('left')
    elif (action == 23):
        pydirectinput.keyDown('right')
        sleep(0.45)
        pydirectinput.keyUp('right')
    # press for 500 miliseconds
    elif (action == 24):
        pydirectinput.keyDown('left')
        sleep(0.5)
        pydirectinput.keyUp('left')
    elif (action == 25):
        pydirectinput.keyDown('right')
        sleep(0.5)
        pydirectinput.keyUp('right')
    #press for 550 miliseconds
    elif (action == 26):
        pydirectinput.keyDown('left')
        sleep(0.55)
        pydirectinput.keyUp('left')
    elif (action == 27):
        pydirectinput.keyDown('right')
        sleep(0.55)
        pydirectinput.keyUp('right')
    # stay still, press neither left nor right
    elif (action == 28):
        pass
    else:
        print("\n--ERROR: action " + str(action) + " selected outside valid action space of " + str(NUM_ACTIONS) + " actions--\n")
    
    return

def end_of_game_status_report(learn_cycles_snuck_in, num_games_completed, score, avg_score, agent, max_time_survived, global_time):
    print("total downtime learning batches: " + str(learn_cycles_snuck_in))
    print("episode", num_games_completed, "| score %.2f" % score,
                        "| average score %.2f" % avg_score,
                        "| epsilon %.4f" % agent.epsilon,
                        "| best score %.2f" % max_time_survived, "seconds ",
                        "| total training time elapsed %.2f" %((time()-global_time)/60), "minutes")

def end_of_game_status_report_test(i, time_survived, avg_time_survived, epsilon, max_time_survived):
    print("episode", i, "| time survived %.2f" % time_survived,
                        "| average time survived %.2f" % avg_time_survived,
                        "| epsilon %.4f" % epsilon,
                        "| best score %.2f" % max_time_survived, "seconds ")

def garbage_collect(agent, debug=False):
    collected = gc.collect()
    if (debug):
        print("Garbage collector: collected", "%d objects." % collected)
        print("agent memory counter:" + str(agent.mem_counter))

def graph_results(x, scores, eps_history, num_games_completed, HOURS_TO_TRAIN, STR_FPS):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("Games Played")
    ax1.set_ylabel("Score")
    ax1.plot(x, scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel("Epsilon")
    ax2.plot(x, eps_history)
    ax2.tick_params(axis='y', labelcolor=color)

    filename = "--.png"
    if (HOURS_TO_TRAIN >= 1):
        time_string = "%.2f" % HOURS_TO_TRAIN
        plt.title("Train Data for " + str(time_string) + " Hours of Training" + "\n(" + str(num_games_completed) + " Games of Super Hexagon)" + "\n(" + STR_FPS + " FPS)")
        time_string = time_string.replace('.', '_')
        filename = "SuperHexagonTrainResults" + time_string + "Hours" + STR_FPS + "FPS.png"
    else: 
        minutes_trained = int(HOURS_TO_TRAIN * 60)
        plt.title("Train Data for " + str(minutes_trained) + " Minutes of Training" + "\n(" + str(num_games_completed) + " Games of Super Hexagon)" + "\n(" + STR_FPS + " FPS)")
        filename = "SuperHexagonTrainResults" + str(minutes_trained) + "Minutes" + STR_FPS + "FPS.png"
    
    plt.show()
    fig.set_size_inches(12, 9)
    fig.savefig(filename)

def graph_results_test(x, scores, eps_history, time_trained_number, time_trained_units, STR_FPS):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel("Games Played")
    ax1.set_ylabel("Time Survived")
    ax1.plot(x, scores, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel("Epsilon")
    ax2.plot(x, eps_history)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Test Data for Model Trained for" + time_trained_number + time_trained_units + "\n(50 Training Games of Super Hexagon)" + "\n(" + STR_FPS + " FPS)")
    filename = "SuperHexagonTestResults5Minutes" + STR_FPS + "FPS.png"
    
    plt.show()
    fig.set_size_inches(12, 9)
    fig.savefig(filename)

def play_count_down():
    print("beginning play in 5...")
    sleep(1)
    print("4...")
    sleep(1)
    print("3...")
    sleep(1)
    print("2...")
    sleep(1)
    print("1...")
    sleep(1)
    print("--- S U P E R   H E X A G O N ---")

def manually_inspect_screenshot_sample(shrink_factor):
    # get shape of the screen, in case you have a weird ultrawide monitor or something
    screenshot = ImageGrab.grab()
    screenshot = np.array(screenshot)
    screen_shape = screenshot.shape
    # shrink it down because my computer is running out of memory
    screenshot_shrunk = cv.resize(screenshot, (0, 0), fx=shrink_factor, fy=shrink_factor)
    screen_shape_shrunk = screenshot_shrunk.shape
    # uncomment for visual inspection of the sample screenshot
    print(screen_shape)
    print(screen_shape_shrunk)
    cv.imshow("original", screenshot)
    cv.imshow("shrunk", screenshot_shrunk)

    # wait 10 seconds
    sleep(10)
    cv.destroyAllWindows()
    return