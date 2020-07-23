[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# BananaCollector
An agent that navigate and collect bananas in a large, square world.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started
1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
2. Place the file in the `unity_simulation/` folder, and unzip (or decompress) the file. 
3. Follow the dependencies guide at the end of this readme (if you haven't allready).
4. Navigate to the project root folder.
```bash
source activate drlnd && jupyter notebook
```
5. Specify the patch to the environment in the first cell of `main.ipynb`.
6. Run the first two cells (*setup.. and train agent*) of `main.ipynb` to start learning.
7. Run the third cell (*score plotter*) to plot the average score against episodes.
8. Run the last cell (*agent viewer*) to show the learned agent.

### Costumization *(optional)*
Turning on and off double Q-learning: set by the in the ***USE_DOUBLE_DQN*** parameter in the top of `banana_agent.py`.

Changeing between sparse and dense network: set by importing either ***dense_DDQN*** or  ***sparse_DDQN*** as  ***DQN*** in the top of  `banana_agent.py`.

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
3. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/MathiasThor/BananaCollector.git
cd BananaCollector/python
pip install .
```

4. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

5. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 


<img src="https://sdk.bitmoji.com/render/panel/e0c28536-e37d-43ff-8a21-2573e0487440-40e9e618-8474-4dc8-a352-f04ad07936f3-v1.png?transparent=1&palette=1 " width="250" height="250">
