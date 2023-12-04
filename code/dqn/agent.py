import numpy as np
import tensorflow as tf

from dqn.replay_buffer import ReplayBuffer


class Agent:
    def __init__(self, Q, Q_target, num_actions, discount_factor=0.99, batch_size=64, epsilon=0.95, epsilon_min=0.05, epsilon_decay=0.995,
                 exploration_type='e-annealing', learning_type='dq', replay_buffer_size=1e5):
        self.Q = Q      
        self.Q_target = Q_target

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.exploration_type = exploration_type
        self.learning_type = learning_type

        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount_factor = discount_factor

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        # start tensorflow session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    # add transition to the replay buffer
    def add(self, state, action, next_state, reward, terminal):
        self.replay_buffer.add_transition(state, action, next_state, reward, terminal)

    # train network
    def train(self):
        # sample batch from the replay buffer
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones = self.replay_buffer.next_batch(self.batch_size)

        # compute td targets using q- or double q-learning
        if self.learning_type == 'q': # q learning
            batch_rewards[np.logical_not(batch_dones)] += self.discount_factor * np.max(self.Q_target.predict(self.sess, batch_next_states), axis=1)[np.logical_not(batch_dones)]
        else: # double q learning
            q_actions = np.argmax(self.Q.predict(self.sess, batch_next_states), axis=1)
            batch_rewards[np.logical_not(batch_dones)] += self.discount_factor * self.Q_target.predict(self.sess, batch_next_states)[np.arange(self.batch_size), q_actions][np.logical_not(batch_dones)]

        # update network and target network
        loss = self.Q.update(self.sess, batch_states, batch_actions, batch_rewards)
        self.Q_target.update(self.sess)

        return loss

    # get action for state
    def act(self, state, deterministic):
        r = np.random.uniform()
        if deterministic or (self.exploration_type != 'boltzmann' and r > self.epsilon):
            # take greedy action (argmax)
            a_pred = self.Q.predict(self.sess, [state])
            action_id = np.argmax(a_pred)
        else:
            if self.exploration_type=='boltzmann':
                actions = self.Q.predict(self.sess, [state])[0]

                # softmax calculation, subtracting max for stability
                actions = np.exp((actions - max(actions)) / self.epsilon)
                actions /= np.sum(actions)

                # selecting action following probabilities
                a_value = np.random.choice(actions, p=actions)
                action_id = np.argmax(a_value == actions)
            else:
                # sample random action
                action_id = np.random.randint(0, self.num_actions)
        return action_id

    # anneal epsilon
    def anneal(self, e=0):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) # linear
        #self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-(1 - self.epsilon_decay) * e))

    # load trained network
    def load(self, folder):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(folder))