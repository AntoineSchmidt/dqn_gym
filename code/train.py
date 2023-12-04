import sys
import gym
import numpy as np
import itertools as it

from utils import *
from dqn.agent import *
from dqn.networks import *
from tensorboard_writer import *


# run one episode in the gym environment
def run_episode(env, agent, deterministic, do_training=True, rendering=True, max_timesteps=1000):
    # save statistics like episode reward or action usage
    stats = EpisodeStats()
    state = env.reset()

    # run environment until max_timesteps or terminal
    for _ in range(max_timesteps):
        action_id = agent.act(state=state, deterministic=deterministic)
        next_state, reward, terminal, info = env.step(action_id)

        # add observation to buffer and train agent
        if do_training:  
            agent.add(state, action_id, next_state, reward, terminal)
            loss = agent.train()

        # save reward to stats
        stats.step(reward, action_id)

        state = next_state

        if rendering:
            env.render()

        if terminal: 
            break

    return stats


# execute the dqn training
def train_online(env, agent, num_episodes, model_dir):
    # create folder for model
    os.makedirs(model_dir, exist_ok=True)

    print("... train agent")

    # setup tensorboard files for monitoring
    tensorboard = TensorboardWriter("train", ["episode_reward", "a_0", "a_1"])
    tensorboard_eval = TensorboardWriter("eval", ["episode_reward", "a_0", "a_1"])

    # training
    for i in range(num_episodes):
        print("episode: ", i)
        stats = run_episode(env, agent, deterministic=False, do_training=True)

        # epsilon anneal
        agent.anneal(i)

        # write data
        tensorboard.write_episode_data(i, eval_dict={"episode_reward" : stats.episode_reward, "a_0" : stats.get_action_usage(0), "a_1" : stats.get_action_usage(1)})

        # check deterministic performance
        if i % 100 == 0 and i != 0:
            stats = run_episode(env, agent, deterministic=True, do_training=False)
            tensorboard_eval.write_episode_data(i, eval_dict={"episode_reward": stats.episode_reward, "a_0": stats.get_action_usage(0), "a_1": stats.get_action_usage(1)})

        # store model every 100 episodes and in the end.
        if i % 100 == 0 or i >= (num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "agent.ckpt"), global_step=i)

    # close tensorboard sessions
    tensorboard.close_session()
    tensorboard_eval.close_session()


if __name__ == "__main__":
    # start tensorboard
    TensorboardWriter.run()

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
        episodes = 1500
        model_dir = "./models_cartpole"
    else:
        env = gym.make("MountainCar-v0").unwrapped
        state_dim = 2
        num_actions = 3
        episodes = 1000
        model_dir = "./models_mountaincar"

    # initialize networks and agent
    q = NeuralNetwork(state_dim, num_actions)
    q_target = TargetNetwork(state_dim, num_actions)
    agent = Agent(q, q_target, num_actions, replay_buffer_size=1e4, epsilon=1.0, epsilon_decay=0.999)

    # train agent
    train_online(env, agent, episodes, model_dir)