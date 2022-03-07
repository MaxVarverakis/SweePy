# https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import random
from itertools import count

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n, m):
        super(DQN, self).__init__()
        self.numActions = n*m
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(1,16,3,1)
        self.relu1 = F.relu()
        self.conv2 = nn.Conv2d(16,32,3,1)
        self.relu2 = F.relu()
        self.fc3 = nn.Linear(1152,188)
        self.relu3 = F.relu()
        self.fc4 = nn.Linear(188,self.numActions)

    def forward(self, x):
        x = x.to(device)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.fc3(x))
        x = x.view(out.size()[0], -1)
        out = self.fc4(x)

        return out

def imTensor(image):
    image = torch.from_numpy(image)
    image = image.to(device)
    return image

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)