import os
import sys
import gym
import json
import numpy as np

from datetime import datetime

from dqn.agent import *
from dqn.networks import *
from train import run_episode


np.random.seed(0)

# read in environment choice
try:
    game = int(sys.argv[1])
except:
    print('Select game: cartpole (1) or mountaincar (default)')
    game = 2

# setup game parameters
if game == 1:
    env = gym.make("CartPole-v0").unwrapped
    state_dim = 4
    num_actions = 2
    model_dir = "./models_cartpole"
else:
    env = gym.make("MountainCar-v0").unwrapped
    state_dim = 2
    num_actions = 3
    model_dir = "./models_mountaincar"

# initialize/load networks and agent
q = NeuralNetwork(state_dim, num_actions)
q_target = TargetNetwork(state_dim, num_actions)
agent = Agent(q, q_target, num_actions)
agent.load(model_dir)

# run number of episodes
n_test_episodes = 15
episode_rewards = []
for i in range(n_test_episodes):
    stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
    episode_rewards.append(stats.episode_reward)

# save results into a .json file
results = {
    "episode_rewards": episode_rewards,
    "mean": np.array(episode_rewards).mean(),
    "std": np.array(episode_rewards).std(),
}

# create folder
if not os.path.exists("./results"):
    os.mkdir("./results")  

# write out results
fname = "./results/cartpole_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
with open(fname, "w") as fh:
    json.dump(results, fh)

# close environment
env.close()