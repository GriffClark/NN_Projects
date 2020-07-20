# AI for self driving car

# IMPORTS
import numpy as np
import random
import os
import torch
import torch.nn as nn  # contains neural network toolkits
import torch.nn.functional
import torch.optim as optim
import torch.autograd


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
        activated_hidden_nodes = torch.nn.functional.relu(self.fc1(state))  # specifying an activation function
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
        return map(lambda x: torch.autograd(torch.cat(x, 0)), samples)


# implementing deep Q learning
class Dqn():
    def __init__(self, input_size, nb_action, gamma):  # input size, num actions, gamma
        self.transitions_in_memory = 100000  # picked this value by trial and error to figure out how much memory we should have
        self.learning_rate = 0.001  # how fast we want the NN to learn
        self.gamma = gamma
        self.reward_window = []  # sliding window to hold rewards
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(self.transitions_in_memory)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # connects optimizer from PyTorch
        # library to model
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # create the tensor framework for our model, and add a
        # fake dimension at index 0
        self.last_action = 0  # will be 0-2 which corresponds to line 33 on map.py (action2rotation)
        self.last_reward = 0  # default reward value

    # need to make the NN select the correct move at each time
    def select_action(self, state):
        # generate a distribution of probabilities based on the inputted q values
        temperature = 7  # the higher this temperature parameter, the more likely we are to choose the highest q action
        probabilities = torch.nn.functional.softmax(
            self.model(torch.autograd(state, volatile=True)) * temperature)  # applies a
        # softmax function to the last state to normalize the output
        action = probabilities.multinomial()  # gives us a random draw from the distribution
        return action.data[0, 0]  # returns 0,1,2 in accordance with action2rotation

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        # transition of the Markov decision process
        outputs = self.model(batch_state).gather((1, batch_action).unsqueeze(1)).squeeze(
            1)  # outputs the chosen action,
        # and kills the fake dimensions
        next_outputs = self.model(batch_next_state).detach().max(1)[0]  # gets qMax for action 0 based on state 1
        target = self.gamma * next_outputs + batch_reward
        td_loss = torch.nn.functional.smooth_l1_loss(outputs, target)  # calculate loss
        self.optimizer.zero_grad()  # have to re-init at each step of gradient decent
        td_loss.backward(retain_variables=True)  # back-propigates loss through the network
        self.optimizer.step()  # this will update the weights using the optimizer

