import numpy as np
import random
from collections import namedtuple, deque

from network import sparse_DDQN as DQN # import either dense_DDQN or sparse_DDQN

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = 1000000   # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.001             # for soft update of target parameters
LR = 0.0003             # learning rate 
UPDATE_EVERY = 4        # how often to update the network
USE_DOUBLE_DQN = True   # use double dqn or not


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    
    def __init__(self, state_size, action_size, seed):
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Initialize Q-Networks
        self.qnetwork_local = DQN(state_size, action_size, seed).to(device)
        self.qnetwork_target = DQN(state_size, action_size, seed).to(device)
        
        # Initialize optimizer 
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Initialize optimizer 
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get memory sample and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)
    
    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) # Get state
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state) # Do forward pass
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # Choose best action
        else:
            return random.choice(np.arange(self.action_size))  # Choose random action
    
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        if USE_DOUBLE_DQN: # Double Q Learning
            self.qnetwork_local.eval()
            with torch.no_grad():
                # Get argmax Q(s,a) using local
                _, Q_actions_next_local = self.qnetwork_local(next_states).max(1, keepdim=True)
                # Use target DQN to calculate Q value
                # With gather we can select the max action values from each output of the target DQN
                Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_actions_next_local)
            self.qnetwork_local.train() 
        else:     
            # Get max predicted Q values (for next states) from target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute MSE loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss using optimizer
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Softly update target Q-network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        # Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    


class ReplayBuffer:
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        # Add a new experience to memory.
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        # Randomly sample a batch of experiences from memory.
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        # Return the current size of internal memory.
        return len(self.memory)
