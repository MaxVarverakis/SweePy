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
import os
import sys
import time
from Minesweeper import minesweeper as ms

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


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

        # Currently configured for 5x5 pixel image inputs
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

def train(model, n, m, mineWeight, start):
	# define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = ms(n, m, mineWeight)

    # initialize replay memory
    memory = ReplayMemory(model.replay_memory_size)

    # initial action
    action = ms.choose()
    img, reward, terminal = game_state.move(action)
    state = imTensor(img)

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

        # get corresponding action from neural network output
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]
        action = (action_index.item() % game_state.m, action_index.item() // game_state.m)

        # get next state and reward
        next_img, _, _ = game_state.move(action)
        next_state = imTensor(next_img)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward, terminal)

        # epsilon decay
        epsilon = epsilon_decay[iteration]
        
        # sample random batch
        transitions = memory.sample(min(model.batch_size,len(memory)))
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Unpack batch
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # get output for the next state
        next_batch = model(next_state_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if batch.terminal[i]
                                  else reward_batch[i] + model.gamma * torch.max(next_batch[i])
                                  for i in range(len(batch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)


        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # compute loss
        loss = criterion(q_value, y_batch)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        iteration += 1

        if iteration % 25000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")
        
        print("iteration: ", iteration, "elapsed time: ", time.time() - start, "epsilon: ", epsilon, "action: ",
              action_index.cpu().detach().numpy(), "reward: ", reward.numpy()[0][0], "Q max: ",
              np.max(output.cpu().detach().numpy()))

def test(model, n, m, mineWeight):
    game_state = ms(n, m, mineWeight)

    # initial action
    action = ms.choose()
    img, _, _ = game_state.move(action)
    state = imTensor(img)

    while True:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = (0,0)
        
        # get corresponding action from neural network output
        action_index = torch.argmax(output)
        action = (action_index.item() % game_state.m, action_index.item() // game_state.m)

        # get next state and reward
        next_img, _, _ = game_state.move(action)
        next_state = imTensor(next_img)

        state = next_state

def main(mode, n, m, mineWeight):
    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_******.pth',
            map_location='cpu').eval()
        
        test(model, n, m, mineWeight)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = DQN(n,m)

        model.apply(init_weights)
        start = time.time()

        train(model, n, m, mineWeight, start)

if __name__ == '__main__':
    # main(sys.argv[1])
    print(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    # main(input('train/test? '))