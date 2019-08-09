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

def run(nthreads, tmax, minibatch_size, nepochs, base_dir='./examples', device='/gpu:0'):

    print("Created agent...")
    agent = ProximalPolicy()

    print("Setting up directories...")
    if base_dir is not None:
        ckpt_dir, log_dir, meta_dir = create_directories('CartPole-v0', "ppo", base_dir)
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

            # perform updates
            for _ in range(nepochs):
                idx = np.random.choice(range(steps), size=minibatch_size)
                summary = agent.update(
                    [states[i] for i in idx],
                    [actions[i] for i in idx],
                    [targets[i] for i in idx],
                    sess
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
    run(nthreads=8, tmax=128, minibatch_size=32*8, nepochs=3)
