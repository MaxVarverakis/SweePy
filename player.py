'''
References:

https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
https://github.com/nevenp/dqn_flappy_bird/blob/master/dqn.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
from itertools import count
import os
import time
from Minesweeper import minesweeper as ms

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('img', 'state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory():

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
        self.batch_size = 32

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

def train(n, m, model, start):
	# define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = ms(n,m)

    # initialize replay memory
    memory = ReplayMemory(model.replay_memory_size)

    # initial action
    action = ms.choose()
    img, action, next_state, reward, terminal = game_state.move(action)
    state = imTensor(img)

    
    # if len(memory) < model.batch_size:
    #     return
    # transitions = memory.sample(model.batch_size)
    # # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # # detailed explanation). This converts batch-array of Transitions
    # # to Transition of batch-arrays.
    # batch = Transition(*zip(*transitions))

    # # Compute a mask of non-final states and concatenate the batch elements
    # # (a final state would've been the one after which simulation ended)
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
    #                                       batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state
    #                                             if s is not None])
    # state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    # reward_batch = torch.cat(batch.reward)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    epsilon_decay = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    iteration = 0

    # def select_action(state):
    #     global steps_done
    #     sample = random.random()
    #     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
    #         math.exp(-1. * steps_done / EPS_DECAY)
    #     steps_done += 1
    #     if sample > eps_threshold:
    #         with torch.no_grad():
    #             # t.max(1) will return largest column value of each row.
    #             # second column on max result is index of where max element was
    #             # found, so we pick action with the larger expected reward.
    #             return policy_net(state).max(1)[1].view(1, 1)
    #     else:
    #         return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

# main infinite loop
    while iteration < model.number_of_iterations:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = (0,0)

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        # if random_action:
        #     print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]
        action = (action_index.item() % game_state.m, action_index.item() // game_state.m)

        # get next state and reward
        next_img, _, _, reward, terminal = game_state.move(action)
        next_state = imTensor(next_img)

        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)
