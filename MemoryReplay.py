from collections import namedtuple, deque
import torch
import random
import numpy as np
from SumTree import SumTree

# if gpu is to be used **make sure to have this file on same computer as agent**
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('state', 'action_idx', 'next_state', 'reward', 'terminal'))

class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        '''Save a transition'''
        self.memory.append(Transition(*args))

    def sample(self, minibatch_size):
        return random.sample(self.memory, minibatch_size)

    def __len__(self):
        return len(self.memory)

    def unpack_minibatch(self, transitions):
        # Transpose the minibatch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts minibatch-array of Transitions
        # to Transition of minibatch-arrays.
        minibatch = Transition(*zip(*transitions))

        # Debugging stuff
        # if iteration == model.number_of_iterations - 1:
        #     print(transitions)

        # Unpack minibatch
        state_minibatch = torch.cat(minibatch.state)
        action_minibatch = torch.cat(minibatch.action_idx)
        reward_minibatch = torch.cat(minibatch.reward)
        next_state_minibatch = torch.cat(minibatch.next_state)
        terminal_minibatch = torch.tensor(minibatch.terminal, device = device, dtype = torch.int)

        return state_minibatch, action_minibatch, reward_minibatch, next_state_minibatch, terminal_minibatch

# PER class from Alex M's repo
class PER():

    def __init__(self, capacity, beta_anneal_steps):
        self.memory_limit = capacity
        # DECREASE ALPHA?
        self.per_alpha = 0.6 # Exponent that determines how much prioritization is used (alpha = 1 => uniform sampling)
        self.per_beta_min = 0.4 # Starting value of importance sampling correction
        self.per_beta_max = 1.0 # Final value of beta after annealing
        self.per_beta_anneal_steps = beta_anneal_steps # Number of steps to anneal beta over
        self.per_epsilon = 0.01 # Small positive constant to prevent zero priority
        
        self.beta_anneal = (self.per_beta_max - self.per_beta_min) / self.per_beta_anneal_steps
        self.per_beta = self.per_beta_min
        self.sumtree = SumTree(capacity)
        self.memory_length = 0

    def unpack_minibatch(self, transitions):
        # Transpose the minibatch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts minibatch-array of Transitions
        # to Transition of minibatch-arrays.
        minibatch = Transition(*zip(*transitions))

        # Debugging stuff
        # if iteration == model.number_of_iterations - 1:
        #     print(transitions)

        # Unpack minibatch
        state_minibatch = torch.cat(minibatch.state)
        action_minibatch = torch.cat(minibatch.action_idx)
        reward_minibatch = torch.cat(minibatch.reward)
        next_state_minibatch = torch.cat(minibatch.next_state)
        terminal_minibatch = torch.tensor(minibatch.terminal, device = device, dtype = torch.int)

        return state_minibatch, action_minibatch, reward_minibatch, next_state_minibatch, terminal_minibatch

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
        
        #Debugging stuff
        # print(self.sumtree.total(), max_weight, np.min(priorities), '\n')
        
        for priority in priorities:
            probability = priority / self.sumtree.total()
            weight = (probability * self.memory_length) ** (-self.per_beta)
            is_weights.append(weight / max_weight)
            
        return minibatch, tree_indices, np.array(is_weights)