import gym
import numpy as np
import matplotlib.pyplot as plt
import sys ; sys.path.append('..')
from Agent import Agent


if __name__ == "__main__":
    # Make enviornment and agent. Notably, epsilon is zeroed out for testing purposes
    env = gym.make("LunarLander-v2")
    joshua = Agent(gamma=0.99, epsilon=0.0, lr=0.0001, input_dims=[8], batch_size=64, n_actions=4, max_mem_size=10000, eps_end=0.005, eps_dec=1e-5)

    # Decide which version of the model you want to test
    # version = "Best"
    version = "Last"
    # num_training_games = "20"
    num_training_games = "500"
    # num_training_games = "800"
    # num_training_games = "1000"
    # num_training_games = "5000"
    # num_training_games = "10000"

    # Load that model into the agent
    joshua.load_models("deepQ"+version+num_training_games+".pt", "targetQ"+version+num_training_games+".pt")
    
    # some things for recording data, and how many games to play
    scores, eps_history = [], []
    n_games = 100
    max_score = 0

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        # Notably, we are not calling learn durring the game
        while not done:
            action = joshua.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            observation = new_observation
        # Data collection on the testing
        scores.append(score)
        eps_history.append(joshua.epsilon)
        avg_score = np.mean(scores[-100:])

        # Just something to watch in real time while we wait. Also
        # lets us know the program hasn't frozen
        print("episode", i+1, "score %2f" % score,
                            "average score %2f" % avg_score,
                            "best score %2f" % max_score)
    
    
    # The rest of this is just plotting the data in a graph and
    # saving it to a .png file
    x =[i+1 for i in range(n_games)]
    
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

    title = "Test Data for " + version + " Model Trained on " + num_training_games + " Games of Lunar Lander\n(Average Score = " + str(avg_score) + ")" + "\n(a score of 200 is considered solved)"
    plt.title(title)
    plt.show()

    filename = "LunarLanderTestResults" + version + num_training_games + ".png"
    fig.set_size_inches(12, 9)
    fig.savefig(filename)
    