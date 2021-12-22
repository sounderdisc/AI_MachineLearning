import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from numpy.core.numerictypes import maximum_sctype


class DeepQNetwork(nn.Module):
    # constructor
    def __init__(self, name, lr, input_dims, n_actions, fc1_dims, fc2_dims, need_CNN=False):
        # call super constructor
        super(DeepQNetwork, self).__init__()
        # initalize instance variables from parameters 
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.need_CNN = need_CNN
        if (need_CNN): self.input_image_channels = input_dims[0]
        # These are the layers in the network, used to define the forward function
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.dimsFromCNN = self.findDimsFromCNN(self.input_dims)
        self.fcCNN = nn.Linear(self.dimsFromCNN, self.fc1_dims) 
        self.conv1 = nn.Conv2d(in_channels=self.input_image_channels, out_channels=15, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=5)
        # These are things concerning training, most are properties inhereited from Module
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=0) # 0<= w_d <=0.1
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        # for saving and loading models
        self.name = name
        self.checkpoint_file = os.path.join(os.getcwd(), self.name+".pt")

    def findDimsFromCNN(self, image_shape):
        if (not self.need_CNN):
            # early, valid integer return in case we aren't using a CNN
            return 42
        else:
            # shape format is [channels, h, w]
            # order of convolutions is 5x5 conv, 2x2 pool, 5x5 conv, 2x2 pool, 5x5 conv
            # ((N-M)/S) for each conv layer
            h = image_shape[1]-5+1
            h = ((h-2)/2)+1
            h = h-5+1
            h = ((h-2)/2)+1
            fin_h = h-5+1

            w = image_shape[2]-5+1
            w = ((w-2)/2)+1
            w = w-5+1
            w = ((w-2)/2)+1
            fin_w = w-5+1
            
            # fin_h = ((image_shape[1]-16) / 4) - 4 # 262 if 1080
            # fin_w = ((image_shape[2]-16) / 4) - 4 # 472 if 1920
            fin_num_channels = 5
            return int(fin_h * fin_w * fin_num_channels) # 618,320 if 1920x108

    # defines the normal, feed foward action of the network
    def forward(self, input):
        if (self.need_CNN):
            # first, 3 convolutional layers, the first two with pooling layers
            x = F.relu(self.conv1(input))
            x = F.max_pool2d(x, kernel_size=2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, kernel_size=2)
            x = F.relu(self.conv3(x))

            # Then, 3 fully connected layers 
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fcCNN(x))
            x = F.relu(self.fc2(x))
            actions = self.fc3(x)

            return actions
        else:
            # simply 3 fully connected layers
            x = F.relu(self.fc1(input))
            x = F.relu(self.fc2(x))
            actions = self.fc3(x)

            return actions

    def save_checkpoint(self, custom_filename="none"):
        print("...saving checkpoint...")
        used_filename = self.checkpoint_file if custom_filename == "none" else custom_filename
        torch.save(self.state_dict(), used_filename)

    def load_checkpoint(self, custom_filename="none"):
        print("...loading checkpoint...")
        used_filename = self.checkpoint_file if custom_filename == "none" else custom_filename
        self.load_state_dict(torch.load(used_filename))


class Agent():
    # constructor
    # gamma - weighting of future rewards
    # epsilon - how often to explore (random action) or exploit (take best known action)
    # lr, input_dims, n_actions - all passed onto the neural network
    # batch_size, max_mem_size - pertain to training 
    # eps_end, eps_dec - the lowest possible epsilon value and how to lower it over time
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=10000, eps_end=0.01, eps_dec=1e-5, need_CNN=False):
        # initalize instance variables from parameters 
        self.gamma = gamma
        self.epsilon = epsilon
        self. eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        # these instance variables control the flow of training
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_counter = 0
        self.learning_steps = 0
        self.target_update_frequency = 3*batch_size
        self.need_CNN = need_CNN

        # instances of class DeepQNetwork, which is defined higher up in this file
        self.deep_Q = DeepQNetwork("deepQ", self.lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, need_CNN=need_CNN)
        self.target_Q = DeepQNetwork("targetQ", self.lr, input_dims, n_actions, fc1_dims=256, fc2_dims=256, need_CNN=need_CNN)

        # These arrays represent the memories the agent forms about game states, what actions it
        # took, and what rewards it got as it plays. Notably, the action memory array's data type
        # is an int, not a float, because the action space must be discrete, not continuous. The
        # terminal memory array is of type bool because we are using it as a mask to tell if a given
        # memory is a terminal memory or not.
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    # at every step of play, the agent will remember what the game state was,
    # what action it took, what reward it got, what the new game state its
    # action put the game into, and if the game ended as a result
    def store_transition(self, state, action, reward, new_state, done):
        # preserving the first 10% of memory helps prevent catostrophic forgetting
        if (self.mem_counter >= self.mem_size):
            self.mem_counter = int(self.mem_size * 0.1)
        index = self.mem_counter
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        # I wish python had a ++ operator. Too bad ints are imutable in python
        self.mem_counter += 1

    def choose_action(self, observation):
        # epsilon decreases over time. If we only do a random action if we roll LOWER than epsilon
        if np.random.random() > self.epsilon:
            state = torch.tensor(observation).to(self.deep_Q.device)
            # dimention problems with CNN functionality
            if (state.dim()<4 and self.need_CNN):
                state = torch.unsqueeze(state, 0).float().to(self.deep_Q.device)
            actions = self.deep_Q.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    # decrement epsilon so we explore less, since we now know a bit more about the game
    def decrement_epsilon(self):
        self.epsilon -= self.eps_dec
        # if epsilon is at min, also decrease the frequency that we update target net
        if (self.epsilon < self.eps_min):
            self.epsilon = self.eps_min
            self.target_update_frequency = 15*self.batch_size
    
    # a target net that changes less frequently helps stablize the model
    def update_target_net(self):
        self.learning_steps += 1
        if (self.learning_steps % self.target_update_frequency == 0):
            self.target_Q.load_state_dict(self.deep_Q.state_dict()) 


    def learn(self):
        # Dont even try to learn if we dont have enough memories yet
        if self.mem_counter < self.batch_size:
            return
        
        # first, zero out gradient so that we are learning only from this batch
        self.deep_Q.optimizer.zero_grad()
        # This gives the highest filled index of our memory. needed in case we have
        # used all of our memory already and our counter is higher than the last index
        max_mem = min(self.mem_counter, self.mem_size)
        # now we need to select a random batch of memories
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        # an array that goes from zero up to the batch size
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # send our data for the selected memory batch to the GPU
        state_batch = torch.tensor(self.state_memory[batch]).to(self.deep_Q.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.deep_Q.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.deep_Q.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.deep_Q.device)

        action_batch = self.action_memory[batch]

        # Now we actually learn from the memory we have selected
        # We are specifically interested in the output for the action we actually took
        q_eval = self.deep_Q.forward(state_batch)[batch_index, action_batch]
        q_next = self.target_Q.forward(new_state_batch)
        # using bool as index is equivalent to if (bool): q_next=0.0
        q_next[terminal_batch] = 0.0
        # the target is the action's true reward considering the max value of the next step's action
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        # calculate loss and backpropagate to update model weights
        loss = self.deep_Q.loss(q_target, q_eval).to(self.deep_Q.device)
        loss.backward()
        self.deep_Q.optimizer.step()

        # bookeeping in order to advance the state of training
        self.decrement_epsilon()
        self.update_target_net()
        
    def save_models(self, custom_filename1="none", custom_filename2="none"):
        self.deep_Q.save_checkpoint(custom_filename1)
        self.target_Q.save_checkpoint(custom_filename2)

    def load_models(self, custom_filename1="none", custom_filename2="none"):
        self.deep_Q.load_checkpoint(custom_filename1)
        self.target_Q.load_checkpoint(custom_filename2)
