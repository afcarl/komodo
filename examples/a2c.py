import time
import json
import os
import gym
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
tf.logging.set_verbosity(tf.logging.ERROR)
from datetime import datetime
from komodo.utils import create_directories, log_scalar, BatchGenerator, NstepBatchGenerator
from komodo.a2c import ActorCritic, NstepActorCritic

def run(nthreads, tmax, base_dir='./examples', device='/gpu:0'):

    print("Created agent...")
    agent = ActorCritic()

    print("Setting up directories...")
    if base_dir is not None:
        ckpt_dir, log_dir, meta_dir = create_directories('CartPole-v0', "a2c", base_dir)
        meta = {
            'env_name': 'CartPole-v0',
        }
        with open(meta_dir + '/meta.json', 'w') as file:
            json.dump(meta, file, indent=2)
    else:
        ckpt_dir = log_dir = None

    print("Starting training...")
    global_steps = 0
    global_episodes = 0
    global_updates = 0
    global_returns = []
    global_start = time.time()
    with tf.Session() as sess:

        # intialize graph
        sess.run(tf.global_variables_initializer())

        # create environments
        envs = [gym.make('CartPole-v0') for _ in range(nthreads)]
        generator = BatchGenerator(envs, agent)

        # setup logging
        writer = tf.summary.FileWriter(log_dir + '/', sess.graph)

        # main loop
        while True:

            # start batch timer
            start = time.time()

            # generate a batch
            data, info = generator.sample(tmax, sess)
            states, actions, targets, episodes, returns, steps = agent.bundle(data, info)

            # perform an update
            summary = agent.update(states, actions, targets, sess)
            fps = steps / (time.time() - start)

            # logging
            global_updates += 1
            global_steps += steps
            global_episodes += episodes
            global_time = time.time() - global_start
            global_returns.extend(returns)
            log_scalar(writer, 'fps', fps , global_steps)
            writer.add_summary(summary, global_steps)
            if episodes > 0:
                if len(global_returns) > 0:
                    avg_return = sum(global_returns[-100:]) / min(len(global_returns), 100)
                else:
                    avg_return = np.nan
                print(f"updates={global_updates}, steps={global_steps}, episodes={global_episodes}, avg_return={avg_return:.2f}, elapsed={global_time:.2f}, fps={fps:.2f}")
                log_scalar(writer, 'avg_return', avg_return, global_steps)
                log_scalar(writer, 'return', np.mean(returns), global_steps)
                if avg_return > 195.0:
                    agent.save(ckpt_dir + "/ckpt", global_steps, sess)
                    print("passed!")
                    break

def run_nstep(nthreads, tmax, base_dir='./examples', device='/gpu:0'):

    print("Created agent...")
    agent = NstepActorCritic()

    print("Setting up directories...")
    if base_dir is not None:
        ckpt_dir, log_dir, meta_dir = create_directories('CartPole-v0', "a2c", base_dir)
        meta = {
            'env_name': 'CartPole-v0',
        }
        with open(meta_dir + '/meta.json', 'w') as file:
            json.dump(meta, file, indent=2)
    else:
        ckpt_dir = log_dir = None

    print("Starting training...")
    global_steps = 0
    global_episodes = 0
    global_updates = 0
    global_returns = []
    global_start = time.time()
    with tf.Session() as sess:

        # intialize graph
        sess.run(tf.global_variables_initializer())

        # create environments
        envs = [gym.make('CartPole-v0') for _ in range(nthreads)]
        generator = NstepBatchGenerator(envs, agent)

        # setup logging
        writer = tf.summary.FileWriter(log_dir + '/', sess.graph)

        # main loop
        while True:

            # start batch timer
            start = time.time()

            # generate a batch
            data, info = generator.sample(tmax, sess)
            states, actions, rewards, next_states, flags = data
            steps, episodes, returns, _ = info

            # perform an update
            summary = agent.update(
                states,
                actions,
                rewards,
                next_states,
                flags,
                sess,
            )
            fps = steps / (time.time() - start)

            # logging
            global_updates += 1
            global_steps += steps
            global_episodes += episodes
            global_time = time.time() - global_start
            global_returns.extend(returns)
            log_scalar(writer, 'fps', fps , global_steps)
            writer.add_summary(summary, global_steps)
            if episodes > 0:
                if len(global_returns) > 0:
                    avg_return = sum(global_returns[-100:]) / min(len(global_returns), 100)
                else:
                    avg_return = np.nan
                print(f"updates={global_updates}, steps={global_steps}, episodes={global_episodes}, avg_return={avg_return:.2f}, elapsed={global_time:.2f}, fps={fps:.2f}")
                log_scalar(writer, 'avg_return', avg_return, global_steps)
                log_scalar(writer, 'return', np.mean(returns), global_steps)
                if avg_return > 195.0:
                    agent.save(ckpt_dir + "/ckpt", global_steps, sess)
                    print("passed!")
                    break


if __name__ == "__main__":
    run_nstep(nthreads=16 * 5, tmax=5)
    # run(nthreads=16, tmax=5)
