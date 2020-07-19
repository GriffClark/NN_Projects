# AI for self driving car

# IMPORTS
import numpy as np
import random
import os
import torch
import torch.nn as nn  # contains neural network toolkits
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as Variable


# creating the architecture of the NN
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()  # python super() syntax ig
        self.input_size = input_size  # size of input layer
        self.nb_action = nb_action  # number of total possible outputs
        hidden_layer_size = 30  # this value is found (at least right now) through experimentation
        # Full Connection 1. Makes the full connection between the input layer and first hidden layer
        self.fc1 = nn.Linear(input_size, hidden_layer_size)
        # Full Connection 2 creates the connections between the hidden layer and the output nodes
        self.fc2 = nn.Linear(hidden_layer_size, nb_action)

    def forward(self, state):
        # applies the activation function to the hidden layer, and returns the output of that hidden layer as q values
        activated_hidden_nodes = F.relu(self.fc1(state))  # specifying an activation function
        q_values = self.fc2(activated_hidden_nodes)
        return q_values


# implementing experience replay
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity  # maximum number of transactions in our history of events
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        # make sure we don't have too much memory
        if len(self.memory) > self.capacity:
            # remove the oldest element to shrink the array
            del self.memory[0]

    def sample(self, batch_size):
        # take some random samples from memory, and reshape it
        samples = zip(*random.sample(self.memory, batch_size))
        # turns this into a torch variable
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

    #im