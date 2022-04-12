from collections import namedtuple, deque
import random
import numpy as np
from SumTree import SumTree

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
        # DECREASE ALPHA?
        self.per_alpha = 0.6 # Exponent that determines how much prioritization is used
        self.per_beta_min = 0.4 # Starting value of importance sampling correction
        self.per_beta_max = 1.0 # Final value of beta after annealing
        self.per_beta_anneal_steps = 50e6 # Number of steps to anneal beta over
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
        
        #Debugging stuff
        # print(self.sumtree.total(), max_weight, np.min(priorities), '\n')
        
        for priority in priorities:
            probability = priority / self.sumtree.total()
            weight = (probability * self.memory_length) ** (-self.per_beta)
            is_weights.append(weight / max_weight)
            
        return minibatch, tree_indices, np.array(is_weights)