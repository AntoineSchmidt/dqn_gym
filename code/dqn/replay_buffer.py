import numpy as np

from collections import namedtuple


# handels the observed tuples (state -> action -> next state and reward)
class ReplayBuffer:
    # setup buffer and maximum size
    def __init__(self, size=1e5):
        self.Data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self.data = self.Data(states=[], actions=[], next_states=[], rewards=[], dones=[])
        self.size = size
        self.index = 0

    # add observed transition to buffer
    def add_transition(self, state, action, next_state, reward, done):
        if len(self.data.states) < self.size:
            self.data.states.append(state)
            self.data.actions.append(action)
            self.data.next_states.append(next_state)
            self.data.rewards.append(reward)
            self.data.dones.append(done)
        else: # fifo if full
            self.data.states[self.index] = state
            self.data.actions[self.index] = action
            self.data.next_states[self.index] = next_state
            self.data.rewards[self.index] = reward
            self.data.dones[self.index] = done
            self.index = int((self.index + 1) % self.size)

    # samples a random batch of transitions
    def next_batch(self, batch_size):
        batch_indices = np.random.choice(len(self.data.states), batch_size)

        batch_states = np.array([self.data.states[i] for i in batch_indices])
        batch_actions = np.array([self.data.actions[i] for i in batch_indices])
        batch_next_states = np.array([self.data.next_states[i] for i in batch_indices])
        batch_rewards = np.array([self.data.rewards[i] for i in batch_indices]).astype(np.float32)
        batch_dones = np.array([self.data.dones[i] for i in batch_indices])

        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones