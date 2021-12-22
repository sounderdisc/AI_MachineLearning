import torch
import torch.nn as nn
import torch.nn.functional as F


class MINSTNeuralNet(nn.Module):
    def __init__(self, mode):
        super(MINSTNeuralNet, self).__init__()
        
        # Define various layers here
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

        self.fcMNISTInFifteenOut = nn.Linear(28*28, 15)
        self.fcFifteenInTenOut = nn.Linear(15, 10)

        self.fcMNISTInThreehundredOut = nn.Linear(28*28, 300)
        self.fcThreehundredInThreehundredOut = nn.Linear(300, 300)
        self.fcThreehundredInTenOut = nn.Linear(300, 10)
        
        # This will select the forward pass function based on mode for the ConvNet.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-3")
            exit(0)
      
    # Baseline sample model
    def model_0(self, X):
        # ======================================================================
        # Three fully connected layers with activation
        
        X = torch.flatten(X, start_dim=1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        X = torch.sigmoid(X)
                
        return X  
        
    # Baseline model. task 1
    def model_1(self, X):
        # Three fully connected layers without activation

        # layer 1, input. flatten function turns the image into a 1d tensor.
        X = torch.flatten(X, start_dim=1)
        # layer 2, hidden layer with no activation
        X = self.fcMNISTInFifteenOut(X)
        # layer 3, output layer with 10 neurons, one for each class. no activation
        X = self.fcFifteenInTenOut(X)
                        
        return X
        

    # task 2
    def model_2(self, X):
        # Train with activation (use model 1 from task 1)
        # Same exact neuron setup as before, but this time we are using ReLU
        # for every layer
        
        X = torch.flatten(X, start_dim=1)
        X = F.relu(X)
        X = self.fcMNISTInFifteenOut(X)
        X = F.relu(X)
        X = self.fcFifteenInTenOut(X)
        X = F.relu(X)
                        
        return X
        

	
    # task 3
    def model_3(self, X):
        # Change number of fully connected layers and number of neurons from model 2 in task 2
        
        X = torch.flatten(X, start_dim=1)
        X = self.fcMNISTInThreehundredOut(X)
        X = F.relu(X)
        X = self.fcThreehundredInThreehundredOut(X)
        X = F.relu(X)
        X = self.fcThreehundredInThreehundredOut(X)
        X = F.relu(X)
        X = self.fcThreehundredInThreehundredOut(X)
        X = F.relu(X)
        X = self.fcThreehundredInThreehundredOut(X)
        X = F.relu(X)
        X = self.fcThreehundredInTenOut(X)
        X = F.relu(X)
        
        return X
        