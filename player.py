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
import time
from Minesweeper import minesweeper as ms

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")

Transition = namedtuple('Transition',
                        ('state', 'action_index', 'next_state', 'reward', 'terminal'))


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
        self.gamma = 0.999
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 100
        self.replay_memory_size = 10000
        self.batch_size = 64

        # Currently tested with 10x10 pixel image inputs
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 3, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(m))
        convh = conv2d_size_out(conv2d_size_out(n))
        linear_input_size = convw * convh * 32
        
        self.fc3 = nn.Linear(linear_input_size, 188)
        self.fc4 = nn.Linear(188, self.numActions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc3(x))
        out = self.fc4(x)

        return out

def imTensor(image):
    # convert image data to tensor and add channel dimension
    image = torch.reshape(torch.from_numpy(image),(1,10,10))
    image = image.to(device)
    # unsqueeze to add batch dimension | final image shape (B,C,H,W) : (1,1,10,10)
    return image.unsqueeze(0).float()

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(model, n, m, mineWeight, start):
	# define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize Huber loss
    criterion = nn.SmoothL1Loss()

    # instantiate game
    game = ms(n, m, mineWeight)

    # initialize replay memory
    memory = ReplayMemory(model.replay_memory_size)

    # initial action
    state = imTensor(game.first_game())

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
        output = model(state)

        # initialize action
        action = (0,0)

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        # if random_action:
        #     print("Performed random action!")

        # get corresponding action from neural network output
        # action_index = [torch.randint(model.numActions, torch.Size([]), dtype=torch.int)
        #                 if random_action
        #                 else torch.argmax(output)][0]
        if random_action:
            action_index = torch.tensor([[random.randrange(model.numActions)]], device = device, dtype = torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_index = output.max(1)[1].view(1, 1)

        action = (action_index.item() % game.cols, action_index.item() // game.cols)
        
        # get next state and reward
        next_img, reward, terminal = game.move(action)
        next_state = imTensor(next_img)
        reward = torch.tensor([reward], device = device, dtype = torch.float32).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action_index, next_state, reward, terminal)

        # epsilon decay
        epsilon = epsilon_decay[iteration]
        
        # sample random batch
        transitions = memory.sample(min(model.batch_size,len(memory)))
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Debugging stuff
        # if iteration == model.number_of_iterations-1:
        #     print(transitions)

        # Unpack batch
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action_index)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        # get output for the next state
        next_output = model(next_state_batch)

        null_reward = torch.tensor([0], device = device, dtype = torch.float32)

        # set y_j to 0 for terminal state or duplicate, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(null_reward if batch.terminal[i]
                                  else reward_batch[i] + model.gamma * torch.max(next_output[i]) if reward_batch[i] != -1
                                  else null_reward
                                  for i in range(len(batch.state))))

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        
        # extract Q-value
        # q_value = torch.sum(model(state_batch) * action_batch)
        q_value = model(state_batch).gather(1, action_batch).squeeze(1)
        
        # Debugging stuff
        # print(np.shape(model(state_batch)),np.shape(action_batch))
        # print(q_value,y_batch)

        # compute loss
        loss = criterion(q_value, y_batch)
        
        # more debugging stuff
        # print(f'Loss: {loss}\nModel: {model(state_batch)}\nAction: {action_batch}\nQ: {q_value}\nY: {y_batch}')
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        iteration += 1

        if iteration % 100000 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")
        
        print("Iteration:", iteration, "\nElapsed time:", time.time() - start, "\nEpsilon:", epsilon, "\nAction:",
              action, "\nReward:", reward.numpy()[0][0], "\nQ max:",
              np.max(output.cpu().detach().numpy()),'\n',
              f'Loss: {loss}')

def test(model, n, m, mineWeight):
    game = ms(n, m, mineWeight)

    # initial action
    state = imTensor(game.first_game())

    while True:
        # get output from the neural network
        output = model(state)

        # initialize action
        action = (0,0)
        
        # get corresponding action from neural network output
        action_index = torch.argmax(output)
        action = (action_index.item() % game.cols, action_index.item() // game.cols)

        # get next state
        next_img, _, terminal = game.move(action)
        next_state = imTensor(next_img)

        state = next_state
        if terminal:
            print(f'Score: {game.found}')
            break
        else:
            print(f'Current Score: {game.found}')

def main(mode, n, m, mineWeight):
    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_100000.pth',
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
    main('train', 10, 10, .175)
    # print(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    # main(input('train/test? '))