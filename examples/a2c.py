import time
import json
import os
import gym
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
tf.logging.set_verbosity(tf.logging.ERROR)
from datetime import datetime
from versterken.utils import create_directories, log_scalar
from versterken.a2c import ActorCritic, BatchGenerator

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
    global_step = 0
    global_episode = 0
    global_update = 0
    global_rewards = []
    start = time.time()
    with tf.Session() as sess:

        # intialize graph
        sess.run(tf.global_variables_initializer())

        # create environments
        envs = [gym.make('CartPole-v0') for _ in range(nthreads)]
        generator = BatchGenerator(envs, agent, sess)

        # setup logging
        writer = tf.summary.FileWriter(log_dir + '/', sess.graph)

        # main loop
        while True:

            # start batch timer
            batch_start = time.time()

            # generate a batch
            batch_data, batch_info = generator.sample(tmax, sess)
            states, actions, rewards, next_states, flags = batch_data
            batch_steps, episodes, episode_rewards, episode_steps = batch_info

            # perform an update
            summary = agent.update(
                states,
                actions,
                rewards,
                next_states,
                flags,
                sess,
                logging=True
            )
            batch_fps = batch_steps / (time.time() - batch_start)

            # logging
            global_update += 1
            global_step += batch_steps
            global_episode += episodes
            global_time = time.time() - start
            global_rewards.extend(episode_rewards)
            log_scalar(writer, 'fps', batch_fps , global_step)
            writer.add_summary(summary, global_step)
            if episodes > 0:
                if len(global_rewards) > 0:
                    avg_return = sum(global_rewards[-100:]) / min(len(global_rewards), 100)
                else:
                    avg_return = np.nan
                print(f"updates={global_update}, steps={global_step}, episodes={global_episode}, avg_return={avg_return:.2f}, elapsed={global_time:.2f}, batch_steps={batch_steps}, batch_size={len(states)}, batch_fps={batch_fps:.2f}")
                log_scalar(writer, 'avg_return', avg_return, global_step)
                log_scalar(writer, 'return', np.mean(episode_rewards), global_step)
                if avg_return > 195.0:
                    agent.save(ckpt_dir + "/ckpt", global_step, sess)
                    print("passed!")
                    break

if __name__ == "__main__":
    run(nthreads=16 * 5, tmax=5)
