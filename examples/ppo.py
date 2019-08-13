import time
import json
import os
import gym
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
tf.logging.set_verbosity(tf.logging.ERROR)
from datetime import datetime
from versterken.utils import create_directories, log_scalar, BatchGenerator
from versterken.ppo import ProximalPolicy

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train', """'Train' or 'test'.""")
tf.app.flags.DEFINE_string('env', 'CartPole-v0', """Gym environment.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, """Initial learning rate.""")
tf.app.flags.DEFINE_float('entropy_beta', 0.01, """Entropy loss coefficient.""")
tf.app.flags.DEFINE_float('value_beta', 0.5, """Value loss coefficient.""")
tf.app.flags.DEFINE_float('gamma', 0.99, """Discount factor in update.""")
tf.app.flags.DEFINE_float('epsilon', 0.1, """Policy loss clipping.""")
tf.app.flags.DEFINE_string('hidden_units', '64,32', """Size of hidden layers.""")
tf.app.flags.DEFINE_string('device', '/gpu:0', """'/cpu:0' or '/gpu:0'.""")
tf.app.flags.DEFINE_integer('nthreads', 8, """Number of environments generating experience.""")
tf.app.flags.DEFINE_integer('tmax', 128, """Maximum trajectory length.""")
tf.app.flags.DEFINE_integer('minibatch_size', 32 * 8, """Examples per training update.""")
tf.app.flags.DEFINE_integer('nepochs', 3, """Update steps per batch.""")
tf.app.flags.DEFINE_float('pass_condition', 195.0, """Average score considered passing environment.""")

if __name__ == "__main__":
    agent = ProximalPolicy(
        env=FLAGS.env,
        lr=FLAGS.learning_rate,
        entropy_beta=FLAGS.entropy_beta,
        value_beta=FLAGS.value_beta,
        gamma=FLAGS.gamma,
        epsilon=FLAGS.epsilon,
        hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],
        device=FLAGS.device
    )
    agent.train(
        nthreads=FLAGS.nthreads,
        tmax=FLAGS.tmax,
        minibatch_size=FLAGS.minibatch_size,
        nepochs=FLAGS.nepochs,
        pass_condition=FLAGS.pass_condition
    )
