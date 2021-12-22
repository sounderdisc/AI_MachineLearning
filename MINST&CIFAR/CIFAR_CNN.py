import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR_CNN(nn.Module):
    def __init__(self, mode):
        super(CIFAR_CNN, self).__init__()
        
        # Define various layers here, such as in the tutorial example
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3)

        self.conv3 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=3)

        # self.conv2 = nn.Conv2d(in_channels=32*32*3, out_channels=10, kernel_size=3) # CIFAR 10 images are 32 by 32 color pixels
        # self.conv3 = nn.Conv2d(in_channels=30*30*10, out_channels=10, kernel_size=3) # (N-F)/S + 1 --> (32-3)/1 + 1 = 30
        
        self.fc1_model1 = nn.Linear(360, 100)  # This is first fully connected layer for step 1.
        self.fc1_model2 = nn.Linear(1440, 100) # This is first fully connected layer for step 2.
        self.fc1_model3 = nn.Linear(640, 100)  # This is first fully connected layer for step 3
        
        self.fc2 = nn.Linear(100, 10)       # This is 2nd fully connected layer for all models.
        
        self.fc_model0 = nn.Linear(2250, 100)   # This is for example model.
        
        
        # This will select the forward pass function based on mode for the ConvNet.
        # Based on the question, you have 3 modes available for step 1 to 3.
        # During creation of each ConvNet model, you will assign one of the valid mode.
        # This will fix the forward function (and the network graph) for the entire training/testing
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 0:
            self.forward = self.model_0
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-3")
            exit(0)
        
    
    # Example model. Modify this for step 1-3
    def model_0(self, X):       
        
        X = F.relu(self.conv1(X))
        #print(X.shape)
        X = F.max_pool2d(X, kernel_size=2)
        #print(X.shape)
        
        X = torch.flatten(X, start_dim=1)
        #print(X.shape)
        
        X = F.relu(self.fc_model0(X))
        X = self.fc2(X)
        
        return X
        
    
    # Simple CNN. step 1
    def model_1(self, X):

        # first convolutional layer
        X = self.conv1(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2) # first pool
        # second convolutional layer
        X = self.conv2(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2) # second pool
     
        # two fully connected layers
        X = torch.flatten(X, start_dim=1) # scaffold forgot to flatten
        X = F.relu(self.fc1_model1(X))
        X = self.fc2(X)
        
        return X
        

    # Increase filters. step 2
    def model_2(self, X):
        
        # first convolutional layer
        X = self.conv3(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2) # first pool
        # second convolutional layer
        X = self.conv4(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2) # second pool
     
        # two fully connected layers
        X = torch.flatten(X, start_dim=1) 
        X = F.relu(self.fc1_model2(X))
        X = self.fc2(X) 
        
        return X
        

    # Large CNN. step 3
    def model_3(self, X):
        
        # first convolutional layer
        X = self.conv3(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2) # first pool
        # second convolutional layer
        X = self.conv4(X)
        X = F.relu(X)
        X = F.max_pool2d(X, kernel_size=2) # second pool
        # third convolutional layer
        X = self.conv5(X)
        X = F.relu(X)
     
        # two fully connected layers
        X = torch.flatten(X, start_dim=1) 
        X = F.relu(self.fc1_model3(X))
        X = self.fc2(X) 
        
        return X
