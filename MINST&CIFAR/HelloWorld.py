# This file is only here so you can make sure you have your environment set up. It will
# give concise errors in the output if you're missing an installed package, and will check if
# your GPU is set up properly with CUDA.

# If you dont wish to run this yourself to watch it train, the output of running both the MINST and CIFAR10
# models is in the results.txt file

# Credit to Dr. Gonzalo Vaca-Castano for the skeleton code this is based on. This was originally the final
# project for his Robot Vision class (CAP 4453) which I expanded on after the class ended.
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter # not used, but I'll leave it here for future convience
import argparse
import numpy as np 


# output = "current directory: " + os.getcwd()
output = "is cuda availible?: " + str(torch.cuda.is_available())

print(output)