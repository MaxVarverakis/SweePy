'''
References:

https://github.com/AlexMGitHub/Minesweeper-DDQN
https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
https://github.com/nevenp/dqn_flappy_bird/blob/master/dqn.py
https://github.com/rlcode/per
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from SumTree import SumTree
from collections import namedtuple, deque
import random
import os
import time
from Minesweeper import minesweeper as ms
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# if gpu is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('state', 'action_idx', 'next_state', 'reward', 'terminal'))

class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(Transition(*args))

    def sample(self, minibatch_size):
        return random.sample(self.memory, minibatch_size)

    def __len__(self):
        return len(self.memory)

# PER class from Alex M's repo
class PER():

    def __init__(self, capacity):
        self.memory_limit = capacity
        self.per_alpha = 0.6 # Exponent that determines how much prioritization is used
        self.per_beta_min = 0.4 # Starting value of importance sampling correction
        self.per_beta_max = 1.0 # Final value of beta after annealing
        self.per_beta_anneal_steps = 10e6 # Number of steps to anneal beta over
        self.per_epsilon = 0.01 # Small positive constant to prevent zero priority
        
        self.beta_anneal = (self.per_beta_max - self.per_beta_min) / self.per_beta_anneal_steps
        self.per_beta = self.per_beta_min
        self.sumtree = SumTree(capacity)
        self.memory_length = 0

    def update_beta(self):
        # Importance sampling exponent beta increases linearly during training
        self.per_beta = min(self.per_beta + self.beta_anneal, self.per_beta_max)

    def push(self, *args):
        '''Save a transition'''
        priority = 1 # Max priority with TD error clipping
        self.sumtree.add(priority, Transition(*args))
        if self.memory_length < self.memory_limit: 
            self.memory_length += 1

    def _per_sample(self, minibatch_size):
        # Implement proportional prioritization according to Appendix B.2.1 
        # of DeepMind's paper "Prioritized Experience Replay"
        minibatch = []
        tree_indices = []
        priorities = []
        is_weights = []

        # Proportionally sample agent's memory        
        samples_per_segment = self.sumtree.total() / minibatch_size
        for segment in range(0, minibatch_size):
            seg_start = samples_per_segment * segment
            seg_end = samples_per_segment * (segment + 1)
            sample = random.uniform(seg_start, seg_end)
            (tree_idx, priority, experience) = self.sumtree.get(sample)
            tree_indices.append(tree_idx)
            priorities.append(priority)
            minibatch.append(experience)
        
        # Calculate and scale weights for importance sampling
        min_probability = np.min(priorities) / self.sumtree.total()
        max_weight = (min_probability * self.memory_length) ** (-self.per_beta)
        for priority in priorities:
            probability = priority / self.sumtree.total()
            weight = (probability * self.memory_length) ** (-self.per_beta)
            is_weights.append(weight / max_weight)
            
        return minibatch, tree_indices, np.array(is_weights)

class DQN(nn.Module):
    def __init__(self, n, m):
        super(DQN, self).__init__()
        self.numActions = n*m
        self.gamma = 0.99
        self.number_of_iterations = 50000
        self.replay_memory_size = 10000
        self.initial_epsilon = 1
        self.final_epsilon = 0.001
        self.epsilon_decay = self.number_of_iterations // 2 # 3
        self.minibatch_scale = 1
        self.minibatch_size = 32 * self.minibatch_scale
        self.target_update = 8 * self.minibatch_scale

        self.pad1 = 0
        self.pad2 = 0
        self.pad3 = 0

        # Currently tested with 10x10 (raw/unpadded) pixel image inputs
        # self.pd = nn.ConstantPad2d(self.pad_size, 225)

        # Alex M Version
        # self.pad1 = 'same'
        # self.pad2 = 'same'
        # self.pad3 = 'same'
        # self.conv1 = nn.Conv2d(9, 64, kernel_size = 5, stride = 1, padding = self.pad1)
        # self.bn1 = nn.minibatchNorm2d(64)
        # self.conv1_5 = nn.Conv2d(64, 64, kernel_size = 5, stride = 1, padding = self.pad1)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = self.pad2)
        # self.bn2 = nn.minibatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 1, kernel_size = 1, stride = 1, padding = self.pad3)

        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5, stride = 1, padding = self.pad1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = self.pad2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 512, kernel_size = 3, stride = 1, padding = self.pad3)
        self.bn3 = nn.BatchNorm2d(512)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        # Will require tweaking if using uneven padding
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(m + 2 * self.pad1, 5) + 2 * self.pad2, 3) + 2 * self.pad3, 3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(n + 2 * self.pad1, 5) + 2 * self.pad2, 3) + 2 * self.pad3, 3)
        linear_input_size = convw * convh * 512
        
        self.fc4 = nn.Linear(linear_input_size, self.numActions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn1(self.conv1_5(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = self.fc4(x)

        return x

def format_state(grid):
    state = torch.tensor(grid, dtype = torch.float, device = device)
    return state.unsqueeze(0)

def imTensor(image):
    # convert image data to tensor and add channel dimension
    image = torch.reshape(torch.from_numpy(image), (1, 10, 10))
    image = image.to(device)
    # unsqueeze to add minibatch dimension | final image shape (B,C,H,W) : (1,1,10,10)
    return image.unsqueeze(0).float()

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def train(n, m, mineWeight, start):

    iteration = 0
    num_episodes = 0

    policy_net = DQN(n, m).to(device)
    policy_net.apply(init_weights)
    target_net = DQN(n, m).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # define Adam optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr = 1e-6)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 1000, verbose = True, factor = 0.66666667)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = policy_net.number_of_iterations // 1.25, gamma = .1, verbose = True)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-5, total_steps = policy_net.number_of_iterations, verbose = True, anneal_strategy = 'cos')

    # initialize loss
    criterion = nn.SmoothL1Loss()

    # instantiate game
    game = ms(n, m, mineWeight)

    # initialize replay memory
    # memory = ReplayMemory(policy_net.replay_memory_size)
    memory = PER(policy_net.replay_memory_size)

    # initial state
    # game.aid()
    # state = format_state(game.one_hot(game.view))
    state = format_state(game.condensed())

    # initialize action
    action = (0,0)

    # initialize loss history
    l = []

    # main infinite loop
    while iteration < policy_net.number_of_iterations:

        # epsilon decay
        epsilon = policy_net.final_epsilon + (policy_net.initial_epsilon - policy_net.final_epsilon) * math.exp(-1. * iteration / policy_net.epsilon_decay)

        # get output from the neural network
        output = policy_net(state)

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        # if random_action:
        #     print("Performed random action!")

        # get corresponding action from neural network output
        # action_idx = [torch.randint(model.numActions, torch.Size([]), dtype=torch.int)
        #                 if random_action
        #                 else torch.argmax(output)][0]
        if random_action:
            action_idx = torch.tensor([[random.randrange(policy_net.numActions)]], device = device, dtype = torch.long)
        else:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_idx = output.max(1)[1].view(1, 1)

        action = (action_idx.item() % game.cols, action_idx.item() // game.cols)

        # get next state and reward
        next_state, reward, terminal = game.move(action)

        next_state = format_state(next_state)
        reward = torch.tensor([reward], device = device, dtype = torch.float32).unsqueeze(0)
        
        # +1 for the number of games played
        if terminal:
            num_episodes += 1

        # Store the transition in memory
        memory.push(state, action_idx, next_state, reward, terminal)
        # memory.push((state, action_idx, next_state, reward, terminal))

        # epsilon decay
        # epsilon = epsilon_decay[iteration]
        
        # sample random minibatch
        # transitions = memory.sample(min(policy_net.minibatch_size, len(memory)))
        transitions, tree_indices, is_weights = memory._per_sample(min(policy_net.minibatch_size, iteration + 1))
        
        # Transpose the minibatch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts minibatch-array of Transitions
        # to Transition of minibatch-arrays.
        minibatch = Transition(*zip(*transitions))

        # Debugging stuff
        # if iteration == model.number_of_iterations-1:
        #     print(transitions)

        # Unpack minibatch
        state_minibatch = torch.cat(minibatch.state)
        action_minibatch = torch.cat(minibatch.action_idx)
        reward_minibatch = torch.cat(minibatch.reward)
        next_state_minibatch = torch.cat(minibatch.next_state)

        # get output for the next state
        next_output = target_net(next_state_minibatch)

        # set y_j = reward if game ends, otherwise y_j = reward + gamma * max(Q)
        y_minibatch = torch.cat(tuple(reward_minibatch[i] + (1 - minibatch.terminal[i]) * policy_net.gamma * torch.max(next_output[i]) for i in range(len(minibatch.state))))

        # null_reward = torch.tensor([0], device = device, dtype = torch.float32)

        # y_minibatch = torch.cat(tuple(
        #                         # set y_j to 0 if stepped on mine or duplicate
        #                         null_reward if reward_minibatch[i] == -1

        #                         # # set y_j to reward if game is won
        #                         # else reward_minibatch[i] if minibatch.terminal[i]

        #                          # set y_j = reward + gamma * max(Q) if stepped on non-mine and set y_j = reward if game is won
        #                         else reward_minibatch[i] + (1 - minibatch.terminal[i]) * policy_net.gamma * torch.max(next_output[i])
                                
        #                         for i in range(len(minibatch.state))))

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_minibatch = y_minibatch.detach()

        
        # extract Q-value
        q_value = policy_net(state_minibatch).gather(1, action_minibatch).squeeze(1)
        
        # Update sum tree with new priorities of sampled experiences
        td_error = q_value - y_minibatch
        td_error = torch.clip(td_error, -1, 1) # Clip for stability
        priority = (torch.abs(td_error) + memory.per_epsilon)  ** memory.per_alpha
        for i in range(min(policy_net.minibatch_size, iteration)):
            memory.sumtree.update(tree_indices[i], priority[i])
        
        # Anneal PER Beta for IS weights
        if iteration >= policy_net.minibatch_size == 0:
            memory.update_beta()

        # Debugging stuff
        # print(np.shape(model(state_minibatch)),np.shape(action_minibatch))
        # print(policy_net(state_minibatch),policy_net(state_minibatch).size())

        # compute loss
        # loss = criterion(q_value, y_minibatch)
        # Apply importance sampling weights during loss calculation
        loss = (torch.FloatTensor(is_weights) * criterion(q_value, y_minibatch)).mean()
        
        l.append(loss.item())
        
        # more debugging stuff
        # l.append(loss.detach().numpy())
        # print(f'Loss: {loss}\nModel: {model(state_minibatch)}\nAction: {action_minibatch}\nQ: {q_value}\nY: {y_minibatch}')
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step(loss)

        state = next_state
        iteration += 1

        print("Iteration:", iteration, "\nElapsed time:", time.time() - start, "\nEpsilon:", epsilon, 
                "\nAction:", action, "\nReward:", reward.cpu().numpy()[0][0], 
                "\nQ max:", np.max(output.cpu().detach().numpy()),
                f'\nLoss: {loss}\n')

        # Update target network if enough games have been played
        if num_episodes % policy_net.target_update == 0:
            num_episodes = 0
            target_net.load_state_dict(policy_net.state_dict())

        # Save model
        if iteration % policy_net.number_of_iterations == 0:
            torch.save(policy_net, f'pretrained_model/current_model_{iteration}.pth')
            
            val = policy_net.number_of_iterations // 20
            window = [val if val % 2 == 1 else val + 1][0]

            sl = savgol_filter(l, window, 3)
            plt.plot(l, alpha = .5)
            plt.plot(sl, c = 'tab:blue')
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()

            

def test(model, n, m, mineWeight):
    game = ms(n, m, mineWeight)

    mines = game.mineCount

    # initial state
    # game.aid()
    # imTensor(game.first_game())
    # state = format_state(game.one_hot(game.view))
    state = format_state(game.condensed())
    
    score = []

    # initialize action
    action = (0,0)

    while True:
        # get output from the neural network
        output = model(state)
        
        # get corresponding action from neural network output
        action_idx = torch.argmax(output)
        action = (action_idx.item() % game.cols, action_idx.item() // game.cols)

        # get next state
        next_state, _, terminal = game.move(action)
        # imTensor(next_img)
        next_state = format_state(next_state) 

        state = next_state

        score.append(game.found - game.num_aided)

        if terminal:
            # print(f'Score: {max(score)}')
            break
        # else:
            # print(f'Current Score: {max(score)}')
        
    return [max(score), mines, game.wins]
        
        # debugging stuff
        # print(game.view)

def main(mode, n, m, mineWeight):
    if mode == 'test':
        model = torch.load(
            f'pretrained_model/current_model_50000.pth',
            map_location = device).eval()
        
        return test(model, n, m, mineWeight)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        # model.apply(init_weights)
        start = time.time()

        return train(n, m, mineWeight, start)

if __name__ == '__main__':

    params = [10, 10, .2]
    
    # main('train', *params)
    
    # Learning rate tests
    # lrr = np.geomspace(1e-7,1e-1,1000)
    # data = []
    # global i

    # for i,lr in enumerate(lrr):
    #     data.append([lr, main('train', *params, lr)])
    # plt.semilogx([x[0] for x in data],[x[1] for x in data])
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Loss')
    # plt.show()

    score = []
    mines = []
    wins = 0
    i = 0
    while i < 5000:
        print(i)
        current_game = main('test', *params)
        score.append(current_game[0])
        mines.append(current_game[1])
        wins += current_game[2]
        i += 1
    idx = np.argmax(score)
    if wins == 0:
        print(f'Average Score: {np.mean(score)}', f'Max Score: {max(score)}/{100 - mines[idx]}')
    else:
        print(f'Average Score: {np.mean(score)}', f'Wins: {wins}')
    smooth_data = savgol_filter(score, 501, 3)
    plt.plot(score, label = '__nolegend__', alpha = .5)
    plt.plot(smooth_data, label = '100 Iteration Moving Average', c = 'tab:blue')
    plt.legend()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()