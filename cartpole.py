
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

tf.placeholder(dtype=tf.float32, shape=[2])

class Agent():
    def __init__(self, lr, dim_state, dim_action, dim_hidden):
        # Hyper parameters
        self.learning_rate_ = lr
        self.dim_state_ = dim_state
        self.dim_action_ = dim_action
        self.dim_hidden_ = dim_hidden

        # Forward pass
        self.input_state_ = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dim_state_], name="state")
        hidden_state = slim.fully_connected(
            self.input_state_, self.dim_hidden_,
            biases_initializer=None, activation_fn=tf.nn.relu)
        self.output_ = slim.fully_connected(
            hidden_state, self.dim_action_,
            biases_initializer=None, activation_fn=tf.nn.softmax)
        self.selected_action_ = tf.argmax(self.output_, 1)

        # Backward: compute loss
        self.reward_ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_ = tf.placeholder(shape=[None], dtype=tf.int32)

    @property
    def input_state(self):
        return self.input_state_

    @property
    def output(self):
        return self.output_

    @property
    def selected_action(self):
        return self.selected_action_

