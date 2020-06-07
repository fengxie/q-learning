#! /usr/bin/python3


import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
from os import path, getcwd


class Agent:
    def __init__(self, lr, dim_state, dim_action, dim_hidden):
        # Hyper parameters
        self.learning_rate_ = lr
        self.dim_state_ = dim_state
        self.dim_action_ = dim_action
        self.dim_hidden_ = dim_hidden

        #############################################
        # Forward pass
        #############################################
        # Input vector: four dimension
        #   - Cart Position: -2.4, 2.4
        #   - Cart Velocity	: -inf, inf
        #   - Pole Angle: -41.8, 41.8
        #   - Pole Velocity At Tip: -inf, inf
        self.input_state_ = tf.placeholder(
            dtype=tf.float32, shape=[None, self.dim_state_], name="state")
        hidden_state_1 = slim.fully_connected(
            self.input_state_, self.dim_hidden_,
            biases_initializer=None, activation_fn=tf.nn.relu)
        hidden_state_2 = slim.fully_connected(
            hidden_state_1, self.dim_hidden_,
            biases_initializer=None, activation_fn=tf.nn.relu)
        self.output_ = slim.fully_connected(
            hidden_state_2, self.dim_action_,
            biases_initializer=None, activation_fn=tf.nn.softmax)
        self.selected_action_ = tf.argmax(self.output_, 1)

        # Backward: compute loss
        self.reward_ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_ = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indices_ = tf.range(0, tf.shape(self.output_)[0]) * tf.shape(self.output_)[1] + self.action_
        self.responsible_outputs_ = tf.gather(tf.reshape(self.output_, [-1]), self.indices_)

        self.loss_ = -tf.reduce_mean(tf.log(self.responsible_outputs_) * self.reward_)

        tvars = tf.trainable_variables()
        self.gradient_holders_ = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders_.append(placeholder)

        self.gradients_ = tf.gradients(self.loss_, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch_ = optimizer.apply_gradients(zip(self.gradient_holders_, tvars))

    @property
    def input_state(self):
        return self.input_state_

    @property
    def output(self):
        return self.output_

    @property
    def selected_action(self):
        return self.selected_action_


gamma = 0.99


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


train_network = False

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # env = gym.make('Pendulum-v0')
    tf.reset_default_graph()  # Clear the Tensorflow graph.

    total_episodes = 1000  # Set total number of episodes to train agent on.
    max_ep = 999
    update_frequency = 5

    myAgent = Agent(1e-2, 4, 2, 16)  # Load the agent.
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    if train_network:
        with tf.Session() as sess:
            sess.run(init)
            total_reward = []
            total_length = []

            gradBuffer = sess.run(tf.trainable_variables())
            for ix, grad in enumerate(gradBuffer):
                gradBuffer[ix] = grad * 0

            for i in range(total_episodes):
                s = env.reset()
                running_reward = 0
                ep_history = []
                for j in range(max_ep):
                    # Probabilistically pick an action given our network outputs.
                    a_dist = sess.run(myAgent.output, feed_dict={myAgent.input_state: [s]})
                    a = np.argmax(a_dist == np.random.choice(a_dist[0], p=a_dist[0]))

                    s1, r, d, _ = env.step(a)  # Get our reward for taking an action given a bandit.
                    ep_history.append([s, a, r, s1])
                    s = s1
                    running_reward += r
                    if j % 10 == 0:
                        env.render()

                    if d == True:
                        # Update the network.
                        ep_history = np.array(ep_history)
                        ep_history[:, 2] = discount_rewards(ep_history[:, 2])
                        feed_dict = {myAgent.reward_: ep_history[:, 2],
                                     myAgent.action_: ep_history[:, 1],
                                     myAgent.input_state: np.vstack(ep_history[:, 0])}
                        grads = sess.run(myAgent.gradients_, feed_dict=feed_dict)
                        for idx, grad in enumerate(grads):
                            gradBuffer[idx] += grad

                        if i % update_frequency == 0 and i != 0:
                            feed_dict = dictionary = dict(zip(myAgent.gradient_holders_, gradBuffer))
                            _ = sess.run(myAgent.update_batch_, feed_dict=feed_dict)
                            for ix, grad in enumerate(gradBuffer):
                                gradBuffer[ix] = grad * 0

                        total_reward.append(running_reward)
                        total_length.append(j)
                        break

                    # Update our running tally of scores.
                if i % 100 == 0:
                    print(np.mean(total_reward[-100:]))

            save_path = saver.save(sess, path.join(getcwd(), "cartpole.ckpt"))
            print("Model saved in path: %s" % save_path)

    else:
        with tf.Session() as sess:
            sess.run(init)
            # Restore variables from disk.
            saver.restore(sess, path.join(getcwd(), "cartpole.ckpt"))

            total_reward = []
            for t in range(100):
                running_reward = 0
                s = env.reset()

                for i in range(5000 * max_ep):
                    a_dist = sess.run(myAgent.output, feed_dict={myAgent.input_state: [s]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    s, r, d, _ = env.step(a)  # Get our reward for taking an action given a bandit.
                    running_reward += r
                    env.render()
                    if d == True:
                        total_reward.append(running_reward)
                        break;

                print("Ended with reward {}".format(running_reward))
            print(np.mean(total_reward))

    env.reset()
