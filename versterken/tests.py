import time
import json
import os
import gym
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
tf.logging.set_verbosity(tf.logging.ERROR)
from datetime import datetime
from versterken.utils import BatchGenerator

agent = ActorCritic()
envs = [gym.make('CartPole-v0') for _ in range(2)]
bg = BatchGenerator(envs, agent)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_data, batch_info = bg.sample(5, sess)
