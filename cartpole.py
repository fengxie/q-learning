
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
