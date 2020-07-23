import torch
import torch.nn as nn
import torch.nn.functional as F


class dense_DDQN(nn.Module):
    """
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Dense Dueling Deep Q network
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """

    def __init__(self, state_size, action_size, seed, fc1_units=2048, fc2_units=1024, fc3_units=512):
        self.state_size = state_size
        self.action_size = action_size
        
        super(dense_DDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        
        self.fc3_adv = nn.Linear(fc2_units, fc3_units)
        self.fc3_val = nn.Linear(fc2_units, fc3_units)

        self.fc4_adv = nn.Linear(fc3_units, self.action_size)
        self.fc4_val = nn.Linear(fc3_units, 1)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        adv = F.relu(self.fc3_adv(x))
        val = F.relu(self.fc3_val(x))
      
        adv = self.fc4_adv(adv)
        val = self.fc4_val(val).expand(x.size(0), self.action_size)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        
        return x
    
class sparse_DDQN(nn.Module):
    """
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Sparse Dueling Deep Q network
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    """

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        self.state_size = state_size
        self.action_size = action_size
        
        super(sparse_DDQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        self.fc2_val = nn.Linear(fc1_units, fc2_units)

        self.fc3_adv = nn.Linear(fc2_units, self.action_size)
        self.fc3_val = nn.Linear(fc2_units, 1)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        
        adv = F.relu(self.fc2_adv(x))
        val = F.relu(self.fc2_adv(x))
      
        adv = self.fc3_adv(adv)
        val = self.fc3_adv(val).expand(x.size(0), self.action_size)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.action_size)
        
        return x

"""
===================================
Extra networks (not in documented)
===================================
"""
class DQNetwork(nn.Module):
    """Deep Q network"""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64, fc3_units=32):
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class SDQNetwork(nn.Module):
    """Sparse Deep Q network"""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):   
        super(SDQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
