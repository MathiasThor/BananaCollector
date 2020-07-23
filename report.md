# **Banana Collector using a Deep Q-Learning Agent**

## Introduction
The work is about training an agent using deep reinforcement learning to navigate and collect bananas in a large, square world. A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the agent's goal is to collect as many yellow bananas as possible while avoiding blue bananas.

## Learning algorithm
For the banana collecting task, a double dueling deep Q-learning algorithm using fixed Q-target and experience replay was used. The algorithm is implemented in three files: `main.ipynb`, `banana_agent.py`, and `network.py`, which are all explained the following. Note that all learning hyperparameters are placed at the end of this section.

### The main loop (`main.ipynb`)
`main.ipynb` implements the unity banana collector environment, an agent, and a deep q-learning loop. In this loop, the agent, implemented in `banana_agent.py`, is used to take actions and update the deep Q-network implemented in `network.py`. The loop makes use of the environment to collect rewards and observations and restart the simulation when necessary. Finally, the scores are collected, which can be used for documentation and feedback to the user.

When training is finished (i.e., reaches a desired average score or times out), the deep Q-network weights are saved to checkpoint.pth. It may later be loaded for inspecting the learned behavior of the banana collector agent using the Agent Viewer at the end of `main.ipynb`.

`main.ipynb` also implements a plotter function, that can plot the scores collected in the deep q-learning loop. For an example of such plots, see the experiments and results section.

Finally, `main.ipynb` implements an agent viewer where the saved weights of the deep Q-network can be loaded and tested on the agent in the banana collector environment. Two models can already be loaded from the `/saved_models` directory

### The agent (`banana_agent.py`)
Banana_agent implements the learning algorithms for updating the DQN. First, it specifies hyperparameters used in updating the DQN's. Then it creates two DQN's using `network.py` - a local and a target DQN. This is in order to do fixed Q-targets, which helps to avoid harmful correlations due to updating a guess with a guess. The agent also implements a replay buffer for experience replay, which helps prevent action values from oscillating or diverging catastrophically.

At every time step, the agent enters the step function where the current experience tuples (S, A, R, S') are added to the replay buffer. At every fourth time step, the agent uses a memory sample from the replay buffer to perform a learning step. A memory sample consists of batch_size randomly sampled experience tuples (S, A, R, S', done).

In the learning step, it uses the sampled experience tuples (S, A, R, S', done) from the memory sample to update the values of the Q-network. First, a loss is computed by:
1.  Calculate the best action for next state using the local Q-network
2.  Calculate the best Q-values for next state using the best action and target Q-network
3.  Calculate Q-values using the best Q-values for next state, current state and target Q-network
4.  Calculate expected Q-values values using the local Q-network
5.  Calculate MSE loss between expected Q-values and target Q-values (TD error)

Note that steps 1. and 2. implements double Q-learning. These are optional and can be disabled in the `banana_agent.py`. Recall that the target Q-network is updated slower than the local Q-network and that we use the local Q-network for acting in the environment.

Next, the MSE loss or TD error is minimized by updating the local Q-network using backpropagation and the Adam optimizer. Finally, the local Q-network is used to softly update the target Q-network by using a target network update frequency parameter, TAU. This is in order to use fixed Q-targets.

The last function of the agent is the act function. An action is chosen by inputting the state to local Q-network and choosing the action with the highest value after running a forward pass. However, the action may also be randomly chosen based on the probability specified by epsilon. This is to help exploration while training.

### The deep Q-network (`network.py`)
In `network.py` we specify the structure of the deep Q-networks (DQN) using PyTorch notation. Currently, it has two different implementations:
1.  A dueling DQN with a dense structure
2.  A dueling DQN with sparse structure

The state-space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Both networks thus have 37 inputs. Both networks furthermore have a fully connected output layer with four outputs which is the number of actions that the agent can take (walk up, down, left, and right).

The dueling DQN with a dense structure has the following network structure:
-   37 inputs neurons
-   2048 hidden neurons*
-   1024 hidden neurons*
-   512 hidden neurons*
-   4 output neurons

The dueling DQN with a dense structure has the following network structure:
-   37 inputs neurons
-   64 hidden neurons*
-   32 hidden neurons*
-   4 output neurons

\* all hidden layers are fully connected with Relu activation functions.

### Hyperparameters

All hyperparameters as taken from [Human-level control through deep reinforcement learning]([http://files.davidqiu.com//research/nature14236.pdf](http://files.davidqiu.com//research/nature14236.pdf)). Note that some parameters are further manually tuned to fit the banana collector task.

| Hyperparametser | Value | Description |
|--|--|--|
| minibatch size | 64 | Number of training cases over which each ADAM update is calculated |
| replay memory size | 1000000 | ADAM updates are sampled from this number of most recent frames |
| discount factor | 0.99 | Discount factor gamma used in the Q-learning update |
| update frequency | 4 | The number of actions selected by the agent between successive ADAM updates.  |
| learning rate | 0.0003 | The learning rate used by the optimizer |
| Target network update frequency | 0.001 | Parameter (tau) for soft updating the target deep Q-network parameters. θ_target = tau*θ_local + (1 - tau)*θ_target |
| initial exploration | 1.0 | The initial value of epsilon in epsilon-greedy exploration |
| final exploration | 0.015 | The final value of epsilon in epsilon-greedy exploration |
| linear exploration decay | 0.0027 | The linear decay value for updating epsilon. epsilon = epsilon - linear_decay. |
| | | |

## Experiments and results

The dueling deep Q-network with a dense structure and sparse structure was tested on the banana collector agent. Both network structure was using dueling, double Q-learning, fixed Q-targets, and experience replay. All hyperparameters used in the tests are specified in the above section. Each network was trained for 1000 episodes in five trials each.

The results from the experiments are shown in the following reward per episode plots. The plot shows that both network structures can achieve an average reward (over 100 episodes) of at least +13 (see dashed green line). The dense network structure archives this within ~420 episodes while the sparse network only needs ~380 episodes. This is presumably due to the fact that fewer weights need to be learned in the spare network. The sparse network is furthermore able to achieve a higher converged score than the dense network.

![plot](https://github.com/MathiasThor/BananaCollector/blob/master/data/score_DQN.png)
*Plot of the score per episode for the dense (blue) and sparce network (orange). The thick darker lines is a rolling mean of the average score per episode. The thin lighter high frequent line is without rolling mean.*

## Future work
In future work, it would be interesting to implement prioritized experience replay as well as fine-tuning the hyperparameters further. It would also be interesting to learn directly from pixels and use convolutional layers in the deep Q-network.

