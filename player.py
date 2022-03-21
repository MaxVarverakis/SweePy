'''
References:

https://github.com/AlexMGitHub/Minesweeper-DDQN
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
import matplotlib.pyplot as plt

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.initial_epsilon = 1
        self.final_epsilon = 0.001
        self.number_of_iterations = 100000
        self.replay_memory_size = 10000
        self.batch_size = 32 # 1024
        self.target_update = 8 # 80
        self.pad_size = 0

        # Currently tested with 10x10 (raw/unpadded) pixel image inputs
        # self.pd = nn.ConstantPad2d(self.pad_size, 225)
        self.conv1 = nn.Conv2d(9, 32, kernel_size = 5, stride = 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 512, kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(512)
        # self.conv4 = nn.Conv2d(512, self.numActions, kernel_size = 1, stride = 1)

        # ADD PADDING WITHIN CONV LAYERS?
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        # Will require tweaking if using uneven padding
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(m + 2 * self.pad_size, 5), 3), 3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(n + 2 * self.pad_size, 5), 3), 3)
        linear_input_size = convw * convh * 512
        
        self.fc4 = nn.Linear(linear_input_size, self.numActions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size()[0], -1)
        out = self.fc4(x)

        return out

def format_state(grid):
    state = torch.tensor(grid, dtype = torch.float)
    return state.unsqueeze(0)

def imTensor(image):
    # convert image data to tensor and add channel dimension
    image = torch.reshape(torch.from_numpy(image), (1, 10, 10))
    image = image.to(device)
    # unsqueeze to add batch dimension | final image shape (B,C,H,W) : (1,1,10,10)
    return image.unsqueeze(0).float()

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(n, m, mineWeight, start):
	
    policy_net = DQN(n, m).to(device)
    policy_net.apply(init_weights)
    target_net = DQN(n, m).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    
    # define Adam optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr = 1e-6)

    # initialize loss
    criterion = nn.SmoothL1Loss()

    # instantiate game
    game = ms(n, m, mineWeight)

    # initialize replay memory
    memory = ReplayMemory(policy_net.replay_memory_size)

    # initial state
    state = format_state(game.one_hot(game.view)) # imTensor(game.first_game())

    # initialize epsilon value
    epsilon = policy_net.initial_epsilon
    epsilon_decay = np.linspace(policy_net.initial_epsilon, policy_net.final_epsilon, policy_net.number_of_iterations)

    iteration = 0
    num_episodes = 0

    # initialize action
    action = (0,0)

    # initialize loss history
    l = []

    # main infinite loop
    while iteration < policy_net.number_of_iterations:
        # get output from the neural network
        output = policy_net(state)

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        # if random_action:
        #     print("Performed random action!")

        # get corresponding action from neural network output
        # action_index = [torch.randint(model.numActions, torch.Size([]), dtype=torch.int)
        #                 if random_action
        #                 else torch.argmax(output)][0]
        if random_action:
            action_index = torch.tensor([[random.randrange(policy_net.numActions)]], device = device, dtype = torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_index = output.max(1)[1].view(1, 1)

        action = (action_index.item() % game.cols, action_index.item() // game.cols)

        # get next state and reward
        next_state, reward, terminal = game.move(action)

        next_state = format_state(next_state) # imTensor(next_img)
        reward = torch.tensor([reward], device = device, dtype = torch.float32).unsqueeze(0)
        
        # +1 for the number of games played
        if terminal:
            num_episodes += 1

        # Store the transition in memory
        memory.push(state, action_index, next_state, reward, terminal)

        # epsilon decay
        epsilon = epsilon_decay[iteration]
        
        # sample random batch
        transitions = memory.sample(min(policy_net.batch_size, len(memory)))
        
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
        next_output = target_net(next_state_batch)

        # set y_j = reward if game ends, otherwise y_j = reward + gamma * max(Q)
        # y_batch = torch.cat(tuple(reward_batch[i] + (1 - batch.terminal[i]) * policy_net.gamma * torch.max(next_output[i]) for i in range(len(batch.state))))

        null_reward = torch.tensor([0], device = device, dtype = torch.float32)

        y_batch = torch.cat(tuple(
                                # set y_j to 0 if stepped on mine or duplicate
                                null_reward if reward_batch[i] == -1

                                # # set y_j to reward if game is won
                                # else reward_batch[i] if batch.terminal[i]

                                 # set y_j = reward + gamma * max(Q) if stepped on non-mine and set y_j = reward if game is won
                                else reward_batch[i] + (1 - batch.terminal[i]) * policy_net.gamma * torch.max(next_output[i])
                                
                                for i in range(len(batch.state))))

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        
        # extract Q-value
        q_value = policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # Debugging stuff
        # print(np.shape(model(state_batch)),np.shape(action_batch))
        # print(policy_net(state_batch),policy_net(state_batch).size())

        # compute loss
        loss = criterion(q_value, y_batch)
        
        l.append([loss.item(),0])
        l[-1][1] = np.mean([x[0] for x in l[max(-100,-len(l)):]])
        
        # more debugging stuff
        # l.append(loss.detach().numpy())
        # print(f'Loss: {loss}\nModel: {model(state_batch)}\nAction: {action_batch}\nQ: {q_value}\nY: {y_batch}')
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state
        iteration += 1
        
        print("Iteration:", iteration, "\nElapsed time:", time.time() - start, "\nEpsilon:", epsilon, "\nAction:",
              action, "\nReward:", reward.cpu().numpy()[0][0], "\nQ max:",
              np.max(output.cpu().detach().numpy()),
              '\n'f'Loss: {loss}\n')

        # Update target network if enough games have been played
        if num_episodes % policy_net.target_update == 0:
            num_episodes = 0
            target_net.load_state_dict(policy_net.state_dict())

        # Save model
        if iteration % policy_net.number_of_iterations == 0:
            torch.save(policy_net, f'pretrained_model/current_model_{iteration}.pth')

            plt.plot([x[0] for x in l], label = '__nolegend__', alpha = .8)
            plt.plot([x[1] for x in l], label = '100 Iteration Moving Average')
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

def test(model, n, m, mineWeight):
    game = ms(n, m, mineWeight)

    # initial state
    state = format_state(game.one_hot(game.view)) # imTensor(game.first_game())
    
    score = []
    
    # initialize action
    action = (0,0)

    while True:
        # get output from the neural network
        output = model(state)
        
        # get corresponding action from neural network output
        action_index = torch.argmax(output)
        action = (action_index.item() % game.cols, action_index.item() // game.cols)

        # get next state
        next_state, _, terminal = game.move(action)
        next_state = format_state(next_state) # imTensor(next_img)

        state = next_state

        score.append(game.found-game.num_aided)

        if terminal:
            # print(f'Score: {max(score)}')
            break
        # else:
            # print(f'Current Score: {max(score)}')
        
    return max(score)
        
        # debugging stuff
        # print(game.view)

def main(mode, n, m, mineWeight):
    if mode == 'test':
        model = torch.load(
            'pretrained_model/current_model_100000.pth',
            map_location = device).eval()
        
        return test(model, n, m, mineWeight)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        # model.apply(init_weights)
        start = time.time()

        train(n, m, mineWeight, start)

if __name__ == '__main__':
    
    params = [10, 10, .175]

    # main('train', *params)
    
    data = []
    mean = []
    i = 0
    while i < 5000:
        print(i)
        data.append(main('test', *params))
        mean.append(np.mean([x for x in data[max(-100,-len(data)):]]))
        i += 1
    print(f'Average Score: {np.mean(data)}')
    plt.plot(data, label = '__nolegend__', alpha = .8)
    plt.plot(mean, label = '100 Iteration Moving Score')
    plt.legend()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()