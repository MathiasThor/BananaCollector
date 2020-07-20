from unityagents import UnityEnvironment
import numpy as np
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt



from dqn_agent import Agent
agent = Agent(state_size=37, action_size=4, seed=0)
print("finished")
