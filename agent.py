'''
References:

https://github.com/AlexMGitHub/Minesweeper-DDQN
https://github.com/pytorch/tutorials/blob/master/intermediate_source/reinforcement_q_learning.py
https://github.com/nevenp/dqn_flappy_bird/blob/master/dqn.py
https://github.com/rlcode/per
https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/blob/master/DuelingDDQN/dueling_ddqn_agent.py
https://github.com/ChuaCheowHuan/reinforcement_learning/tree/master/DQN_variants/duel_DDQN_PER
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from Minesweeper import minesweeper as ms
import MemoryReplay as mr

# if gpu is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DDQN(nn.Module):
    def __init__(self, n, m):
        super(DDQN, self).__init__()
        self.numActions = n * m
        self.gamma = 0.99
        self.number_of_iterations = 100000000
        self.replay_memory_size = 100000
        self.initial_epsilon = 1
        self.final_epsilon = 0.001
        self.epsilon_decay = self.number_of_iterations // 1.5
        self.minibatch_scale = 2
        self.minibatch_size = 32 * self.minibatch_scale
        self.target_update = 8 * self.minibatch_scale

        self.pad1 = 1
        self.pad2 = 1
        self.pad3 = 1
        # self.pad4 = 1
        # self.pad5 = 0

        # self.pd = nn.ConstantPad2d(self.pad_size, 225)

        # Alex M Version
        # self.pad1 = 'same'
        # self.pad2 = 'same'
        # self.pad3 = 'same'
        # self.bn = nn.BatchNorm2d(64)
        # self.conv1 = nn.Conv2d(2, 64, kernel_size = 5, stride = 1, padding = self.pad1)
        # # self.conv1_5 = nn.Conv2d(64, 64, kernel_size = 5, stride = 1, padding = self.pad1)
        # self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = self.pad2)
        # self.conv3 = nn.Conv2d(64, 1, kernel_size = 1, stride = 1, padding = self.pad3)
        
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

        self.conv1 = nn.Conv2d(2, 64, kernel_size = 3, stride = 1, padding = self.pad1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = self.pad2, bias = False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, kernel_size = 3, stride = 1, padding = self.pad3, bias = False)
        self.bn3 = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(64, 512, kernel_size = 3, stride = 1, padding = self.pad4)
        # self.bn4 = nn.BatchNorm2d(512)
        # self.conv5 = nn.Conv2d(128, 512, kernel_size = 3, stride = 1, padding = self.pad5)
        # self.bn5 = nn.BatchNorm2d(512)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size, stride = 1):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        # # Will require tweaking if using uneven padding
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(m + 2 * self.pad1, 3) + 2 * self.pad2, 3) + 2 * self.pad3, 3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(n + 2 * self.pad1, 3) + 2 * self.pad2, 3) + 2 * self.pad3, 3)
        # convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(m + 2 * 1, 3) + 2 * 1, 3) + 2 * 1, 3)
        # convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(n + 2 * 1, 3) + 2 * 1, 3) + 2 * 1, 3)
        linear_input_size = convw * convh * 256
        
        self.fc1 = nn.Linear(linear_input_size, self.numActions)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(x.size()[0], -1)
        # x = self.fc1(x)
        x = self.fc1(x)

        # x = F.relu(self.bn(self.conv1(x)))
        # x = F.relu(self.bn(self.conv2(x)))
        # x = F.relu(self.bn(self.conv2(x)))
        # x = F.relu(self.bn(self.conv2(x)))
        # x = F.relu(self.bn(self.conv2(x)))
        # x = F.relu(self.bn(self.conv2(x)))
        # x = F.relu(self.bn(self.conv2(x)))
        # x = self.conv3(x)
        # x = x.view(x.size()[0], -1)

        return x

def format_state(grid):
    state = torch.tensor(np.array(grid), dtype = torch.float, device = device)
    return state.unsqueeze(0)

def imTensor(image):
    # convert image data to tensor and add channel dimension
    image = torch.reshape(torch.from_numpy(image), (1, 10, 10), device = device)
    image = image
    # unsqueeze to add minibatch dimension | final image shape (B,C,H,W) : (1,1,10,10)
    return image.unsqueeze(0).float()

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.uniform_(m.weight, -0.01, 0.01)
        # m.bias.data.fill_(0.01)

def train(n, m, mineWeight, start):

    iteration = 0
    num_episodes = 0
    total_episodes = 1

    policy_net = DDQN(n, m).to(device)
    policy_net.apply(init_weights)
    target_net = DDQN(n, m).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # define Adam optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr = 1e-6)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = policy_net.number_of_iterations // 3.1, gamma = .5, verbose = True)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 1000, verbose = True, factor = 0.66666667)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-5, total_steps = policy_net.number_of_iterations, verbose = True, anneal_strategy = 'cos')

    # initialize loss
    criterion = nn.SmoothL1Loss()

    # instantiate game
    game = ms(n, m, mineWeight)

    # initialize replay memory
    # memory = mr.ReplayMemory(policy_net.replay_memory_size)
    memory = mr.PER(policy_net.replay_memory_size, policy_net.number_of_iterations // 1.15)

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
                # action_idx = torch.argmax(A)

        action = (action_idx.item() % game.cols, action_idx.item() // game.cols)

        # get next state and reward
        next_state, reward, terminal = game.move(action)

        next_state = format_state(next_state)
        reward = torch.tensor([reward], device = device, dtype = torch.float32).unsqueeze(0)
        
        # +1 for the number of games played
        if terminal:
            total_episodes += 1
            num_episodes += 1

        # Store the transition in memory
        memory.push(state, action_idx, next_state, reward, terminal)

        # epsilon decay
        # epsilon = epsilon_decay[iteration]
        
        if iteration < policy_net.minibatch_size:
            iteration += 1
            continue
        # sample random minibatch
        # transitions = memory.sample(min(policy_net.minibatch_size, len(memory)))
        transitions, tree_indices, is_weights = memory._per_sample(min(policy_net.minibatch_size, iteration + 1))
        
        # Transpose the minibatch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts minibatch-array of Transitions
        # to Transition of minibatch-arrays.
        # minibatch = Transition(*zip(*transitions))

        # Debugging stuff
        # if iteration == model.number_of_iterations - 1:
        #     print(transitions)

        # Unpack minibatch
        # state_minibatch = torch.cat(minibatch.state)
        # action_minibatch = torch.cat(minibatch.action_idx)
        # reward_minibatch = torch.cat(minibatch.reward)
        # next_state_minibatch = torch.cat(minibatch.next_state)
        # terminal_minibatch = torch.tensor(minibatch.terminal, device = device, dtype = torch.int)

        state_minibatch, action_minibatch, reward_minibatch, next_state_minibatch, terminal_minibatch = memory.unpack_minibatch(transitions)

        # get output for the next state
        next_output = target_net(next_state_minibatch)
        
        # get outputs for the Q calculation
        # V, A = policy_net(state_minibatch)
        # V_next, A_next = target_net(next_state_minibatch)
        # V_eval, A_eval = policy_net(next_state_minibatch)
        
        # set y_j = reward if game ends, otherwise y_j = reward + gamma * max(Q)
        y_minibatch = torch.cat(tuple(
            reward_minibatch[i] + torch.logical_not(terminal_minibatch[i]).int() * policy_net.gamma * torch.max(next_output[i])
            for i in range(len(state_minibatch))))

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_minibatch = y_minibatch.detach()

        
        # extract Q-value
        q_value = policy_net(state_minibatch).gather(1, action_minibatch).squeeze(1)
        
        # indices = np.arange(policy_net.minibatch_size)
        # # q_pred = torch.add(V, (A - A.mean(dim = 1, keepdim = True)))
        # # print(q_pred, q_pred.size(),[indices, action_minibatch])

        # q_pred = torch.add(V, (A - A.mean(dim = 1, keepdim = True)))[indices, action_minibatch]
        # q_next = torch.add(V_next, (A_next - A_next.mean(dim = 1, keepdim = True)))
        # q_eval = torch.add(V_eval, (A_eval - A_eval.mean(dim = 1, keepdim = True)))


        # max_actions = torch.argmax(q_eval, dim = 1)

        # q_target = reward_minibatch + torch.logical_not(terminal_minibatch).int() * policy_net.gamma * q_next[indices, max_actions]
        # # print(terminal, torch.logical_not(terminal).int())
        # # q_value = reward_minibatch + (1 - minibatch.terminal) * policy_net.gamma * torch.max(q_next)

        # # loss = criterion(q_target, q_pred)

        # q_value = A.gather(1, action_minibatch).squeeze(1)

        # Update sum tree with new priorities of sampled experiences
        td_error = q_value - y_minibatch
        # td_error = q_target - q_pred
        # td_error = q_target - q_value
        abs_td_error = torch.clip(td_error, memory.per_epsilon, 1) # Clip for stability
        priority = abs_td_error ** memory.per_alpha
        for i in range(policy_net.minibatch_size):
            # print(priority.size())
            memory.sumtree.update(tree_indices[i], priority[i].item())
        
        # Anneal PER Beta for IS weights
        if iteration >= policy_net.minibatch_size == 0:
            memory.update_beta()

        # Debugging stuff
        # print(np.shape(model(state_minibatch)),np.shape(action_minibatch))
        # print(policy_net(state_minibatch),policy_net(state_minibatch).size())

        # compute loss
        # loss = criterion(q_value, y_minibatch)
        # Apply importance sampling weights during loss calculation
        loss = (torch.FloatTensor(is_weights).to(device) * criterion(q_value, y_minibatch).to(device)).mean()
        # loss = (torch.FloatTensor(is_weights).to(device) * criterion(q_target, q_pred)).mean()
        
        l.append(loss.item())
        
        # more debugging stuff
        # l.append(loss.detach().numpy())
        # print(f'Loss: {loss}\nModel: {model(state_minibatch)}\nAction: {action_minibatch}\nQ: {q_value}\nY: {y_minibatch}')
        
        # backward pass
        for param in policy_net.parameters():
            param.grad = None
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        state = next_state
        iteration += 1

        elapsed = time.time() - start
        time_passed = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        print("Iteration:", iteration, 
            f'\nElapsed time: {time_passed}', 
            "\nEpsilon:", epsilon, 
            "\nTotal Episodes:", total_episodes, 
            # "\nAction:", action, 
            "\nReward:", reward.cpu().numpy()[0][0], 
            "\nQ max:", np.max(output.cpu().detach().numpy()),
            f'\nLoss: {loss}\n')
        # print(scheduler.get_last_lr())

        # Update target network if enough games have been played
        if num_episodes % policy_net.target_update == 0:
            num_episodes = 0
            target_net.load_state_dict(policy_net.state_dict())

        # Save model/Plot data
        if iteration % policy_net.number_of_iterations == 0:
            torch.save(policy_net, f'pretrained_model/current_model_{iteration}.pth')
            
            val = policy_net.number_of_iterations // 20
            window = [val if val % 2 == 1 else val + 1][0]

            sl = savgol_filter(l, window, 3) # moving average
            plt.plot(l, alpha = .5)
            plt.plot(sl, c = 'tab:blue')
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.show()

            

def test(model, n, m, mineWeight, watch = False):
    game = ms(n, m, mineWeight)

    if watch:
        print(game.grid)

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
        # action_idx = A.max(1)[1].view(1, 1).item()
        action = (action_idx.item() % game.cols, action_idx.item() // game.cols)

        # get next state
        next_state, _, terminal = game.move(action)
        # imTensor(next_img)
        next_state = format_state(next_state) 

        state = next_state

        score.append(game.found - game.num_aided)

        if watch:
            print(game.view, action)

        if terminal:
            # print(f'Score: {max(score)}')
            break
        # else:
            # print(f'Current Score: {max(score)}')
        
    return [max(score), mines, game.wins]
        
        # debugging stuff
        # print(game.view)

def main(mode, n, m, mineWeight, watch = False):
    if mode == 'test':
        model = torch.load(
            f'pretrained_model/current_model_100000000.pth',
            map_location = device).eval()
        
        return test(model, n, m, mineWeight, watch)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        # model.apply(init_weights)
        start = time.time()

        return train(n, m, mineWeight, start)

if __name__ == '__main__':

    params = [5, 5, .2]
    
    main('train', *params)
    
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

    # main('test', *params, watch = True)

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
    plt.plot(score, alpha = .5)
    plt.plot(smooth_data, c = 'tab:blue')
    # plt.legend()
    plt.xlabel('Game')
    plt.ylabel('Score')
    plt.show()