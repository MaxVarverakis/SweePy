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
        self.final_epsilon = 0.0001
        self.initial_epsilon = 1
        self.number_of_iterations = 1000000
        self.replay_memory_size = 10000
        self.batch_size = 32
        self.target_update = 10

        # Currently tested with 10x10 pixel image inputs
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 64, kernel_size = 3, stride = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size = 3, stride = 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size = 3, stride = 1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(m, 3), 3), 3), 3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(n, 3), 3), 3), 3)
        linear_input_size = convw * convh * 512
        
        self.fc5 = nn.Linear(linear_input_size, 188)
        self.fc6 = nn.Linear(188, self.numActions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc5(x))
        out = self.fc6(x)

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

def train(n, m, mineWeight, start):
	
    policy_net = DQN(n, m).to(device)
    policy_net.apply(init_weights)
    target_net = DQN(n, m).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    
    # define Adam optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr = 1e-6)

    # initialize Huber loss
    criterion = nn.SmoothL1Loss()

    # instantiate game
    game = ms(n, m, mineWeight)

    # initialize replay memory
    memory = ReplayMemory(policy_net.replay_memory_size)

    # initial state
    state = imTensor(game.first_game())

    # initialize epsilon value
    epsilon = policy_net.initial_epsilon
    epsilon_decay = np.linspace(policy_net.initial_epsilon, policy_net.final_epsilon, policy_net.number_of_iterations)

    iteration = 0
    num_episodes = 0

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
    while iteration < policy_net.number_of_iterations:
        # get output from the neural network
        output = policy_net(state)

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
            action_index = torch.tensor([[random.randrange(policy_net.numActions)]], device = device, dtype = torch.long)
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

        null_reward = torch.tensor([0], device = device, dtype = torch.float32)

        y_batch = torch.cat(tuple(
                                # set y_j to 0 if stepped on mine or duplicate
                                null_reward if reward_batch[i] == -1

                                # set y_j to reward if game is won
                                else reward_batch[i] if batch.terminal[i]

                                 # set y_j = reward + gamma * max(Q) if stepped on non-mine
                                else reward_batch[i] + policy_net.gamma * torch.max(next_output[i])
                                
                                for i in range(len(batch.state))))

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        
        # extract Q-value
        q_value = policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # Debugging stuff
        # print(np.shape(model(state_batch)),np.shape(action_batch))
        # print(q_value,y_batch)

        # compute loss
        loss = criterion(q_value, y_batch)
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
              action, "\nReward:", reward.numpy()[0][0], "\nQ max:",
              np.max(output.cpu().detach().numpy()),
              '\n'f'Loss: {loss}\n')

        # Update target network if enough games have been played
        if num_episodes % policy_net.target_update == 0:
            num_episodes = 0
            target_net.load_state_dict(policy_net.state_dict())

        # Save model
        if iteration % policy_net.number_of_iterations == 0:
            torch.save(policy_net, f'pretrained_model/current_model_{iteration}.pth')
            # steps = [i for i in range(model.number_of_iterations)]
            # plt.plot(steps, l)
            # plt.subplot(1,3,1)
            # plt.plot(steps,rewards_history)
            # plt.xlabel("Iteration")
            # plt.ylabel("Loss")
            # plt.show()
            
            # plt.subplot(1,3,2)
            # plt.plot(steps,q_history)
            # plt.xlabel("Iteration")
            # plt.ylabel("Q max")

            # plt.subplot(1,3,3)
            # plt.plot(steps,q_history)
            # plt.xlabel("Iteration")
            # plt.ylabel("Loss")
            # plt.show()


def test(model, n, m, mineWeight):
    game = ms(n, m, mineWeight)

    # initial state
    state = imTensor(game.first_game())
    
    score = []

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
            'pretrained_model/current_model_1000000.pth',
            # 'pretrained_model/1000000NonZero_NonTermDup.pth',
            map_location='cpu').eval()
        
        return test(model, n, m, mineWeight)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        # model = DQN(n,m).to(device)

        # model.apply(init_weights)
        start = time.time()

        train(n, m, mineWeight, start)

if __name__ == '__main__':
    # data = []
    # mean = []
    # iters = 10000
    # for i in range(iters):
    #     print(i)
    #     data.append(main('test', 10, 10, .175))
    #     mean.append(np.mean(data))
    # plt.plot([j for j in range(iters)], data, label = '__nolegend__')
    # plt.plot([j for j in range(iters)], mean, label = 'Avg Score')
    # plt.legend()
    # plt.xlabel('Game')
    # plt.ylabel('Score')
    # plt.show()
    # print(np.mean(data))

    main('train', 10, 10, .175)