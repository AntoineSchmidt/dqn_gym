import numpy as np
import tensorflow as tf


# Q-Network
class NeuralNetwork:
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4):
        self._build_model(state_dim, num_actions, hidden, lr)
        
    def _build_model(self, state_dim, num_actions, hidden, lr):
        # placeholder
        self.states_ = tf.placeholder(tf.float32, shape=[None, state_dim])
        self.actions_ = tf.placeholder(tf.int32, shape=[None])
        self.targets_ = tf.placeholder(tf.float32,  shape=[None])

        # network
        fc1 = tf.layers.dense(self.states_, hidden, tf.nn.relu)
        fc2 = tf.layers.dense(fc1, hidden, tf.nn.relu)
        self.predictions = tf.layers.dense(fc2, num_actions)

        # get the predictions for the chosen actions only
        batch_size = tf.shape(self.states_)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # calculate the loss
        self.losses = tf.squared_difference(self.targets_, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # setup adam optimizer
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss)

    # predict action rewards for states
    def predict(self, sess, states):
        prediction = sess.run(self.predictions, { self.states_: states })
        return prediction

    # train network on transition batch
    def update(self, sess, states, actions, targets):
        feed_dict = { self.states_: states, self.targets_: targets, self.actions_: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


# slowly updated target network
class TargetNetwork(NeuralNetwork):
    def __init__(self, state_dim, num_actions, hidden=20, lr=1e-4, tau=0.01):
        super().__init__(state_dim, num_actions, hidden, lr)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):
        tf_vars = tf.trainable_variables()
        total_vars = len(tf_vars)
        op_holder = []
        for idx,var in enumerate(tf_vars[0:total_vars//2]):
            op_holder.append(tf_vars[idx+total_vars//2].assign(
              (var.value()*self.tau) + ((1-self.tau)*tf_vars[idx+total_vars//2].value())))
        return op_holder

    # update target network toward network
    def update(self, sess):
        for op in self._associate:
          sess.run(op)