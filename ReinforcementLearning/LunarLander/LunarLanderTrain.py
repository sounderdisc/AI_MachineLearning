import gym
import numpy as np
import matplotlib.pyplot as plt
import sys ; sys.path.append('..')
from Agent import Agent


if __name__ == "__main__":
    # Make enviornment and agent. The agent is named joshua because of the moview WarGames
    env = gym.make("LunarLander-v2")
    joshua = Agent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=[8], batch_size=64, n_actions=4, max_mem_size=10000, eps_end=0.005, eps_dec=1e-5)
    
    # This is a unique approach to dealing with catastophic forgetting. The idea
    # is to simply reset to a prior saved network any time we catasrophically forget
    strong_arm_remember = True
    
    # some things for recording data, and how many games to play
    scores, eps_history = [], []
    n_games = 500
    max_score = 0

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        # The open AI gym game/training cycle involves the agent picking an action, stepping
        # the environment, letting the agent record the effects of their action and perform a
        # batch of learning, then set the new game state as the next step's current game state
        while not done:
            action = joshua.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            joshua.store_transition(observation, action, reward, new_observation, done)
            joshua.learn()
            observation = new_observation
        # Data collection on the training
        scores.append(score)
        eps_history.append(joshua.epsilon)
        avg_score = np.mean(scores[-100:])

        # Here, we are checking to see if we've gotten a new high score with this model, and if we
        # have, we save it to come back to in the last 10% of training games in the event that the
        # model catastrophically forgets. We also save the last model we obtained through normal
        # training before switching over to Strong Arm Remembering
        if (score > max_score or (max_score == 0 and i == n_games*0.9)):
            max_score = score
            joshua.save_models("deepQBest" + str(n_games) + ".pt", "targetQBest" + str(n_games) + ".pt")
        if(strong_arm_remember and i == n_games*0.9):
            joshua.save_models("deepQLast" + str(n_games) + ".pt", "targetQLast" + str(n_games) + ".pt")
        if (strong_arm_remember and max_score-score > 150 and i > n_games*0.9):
            joshua.load_models("deepQBest" + str(n_games) + ".pt", "targetQBest" + str(n_games) + ".pt")

        # Just something to watch in real time while we wait. Also
        # lets us know the program hasn't frozen
        print("episode", i+1, "of", n_games, "score %2f" % score,
                            "average score %2f" % avg_score,
                            "epsilon %2f" % joshua.epsilon,
                            "best score %2f" % max_score)
    
    # Graph x, the games, against the scores and the eps_history
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

    plt.title("Train Data for " + str(n_games) + " Games of Lunar Lander")
    plt.show()

    filename = "LunarLanderTrainResults" + str(n_games) + "Games.png"
    fig.set_size_inches(12, 9)
    fig.savefig(filename)
    