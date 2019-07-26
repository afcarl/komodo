import time
import json
import os
import gym
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress all messages
tf.logging.set_verbosity(tf.logging.ERROR)
from datetime import datetime
from versterken.a2c import ActorCritic
from versterken.utils import create_directories
from versterken.atari import rgb_to_grayscale, collect_frames


def mlp(x, sizes, activation, output_activation=None):
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activation)
    if sizes[-1] == 1:
        return tf.squeeze(tf.layers.dense(x, units=sizes[-1], activation=output_activation))
    else:
        return tf.layers.dense(x, units=sizes[-1], activation=output_activation)

def cnn(x, action_dim, scope=''):

    with tf.variable_scope(scope):

        # conv1
        x = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=8,
            strides=4,
            padding='valid',
            activation=tf.nn.relu)

        # conv2
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=4,
            strides=2,
            padding='valid',
            activation=tf.nn.relu)

        # conv3
        x = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='valid',
            activation=tf.nn.relu)

        # dense
        x = tf.layers.dense(tf.reshape(x, (-1, 64 * 7 * 7)), 512, tf.nn.relu)

        # logits
        return tf.layers.dense(x, action_dim)

def run_episode(env, agent, sess, writer, global_step, render=False):
    episode_return = 0
    episode_steps = 0
    start_step = global_step
    states = []
    actions = []
    rewards = []
    state = env.reset()
    if render == True:
        env.render()
    start_time = time.time()
    while True:
        # perform an action
        action = agent.action(state.reshape(1, -1), sess)
        next_state, reward, done, info = env.step(action)
        if render == True:
            env.render()
        # record transition
        states += [state]
        actions += [action]
        rewards += [reward]
        state = next_state
        # update episode totals
        episode_return += reward
        episode_steps += 1
        # update global step count
        global_step += 1
        if episode_steps % agent.update_freq == 0 or done:
            # update networks
            targets = agent.targets(state.reshape(1, -1), rewards, done, sess)
            summary = agent.update(states, actions, targets, sess)
            writer.add_summary(summary, global_step)
            # reset arrays
            states.clear()
            actions.clear()
            rewards.clear()
        if done:
            break
    episode_fps = (global_step - start_step) / (time.time() - start_time)
    return episode_return, episode_steps, episode_fps, global_step

def run_atari(env, agent, sess, writer, global_step, render=False):
    obs = env.reset()
    obs_queue = Queue(init_values=[preprocess(obs, sess)], size=agent_history)
    state = collect_frames(obs_queue, nframes=agent_history)
    episode_return = 0
    episode_steps = 0
    start_time = time.time()
    start_step = global_step
    states = []
    actions = []
    rewards = []
    if render == True:
        env.render()
    while True:
        # perform an action
        action = agent.action(state.reshape(1, -1), sess)
        obs, reward, done, info = env.step(action)
        if render == True:
            env.render()
            time.sleep(delay)
        # record transition
        states += [state]
        actions += [action]
        rewards += [reward]
        # calculate new state
        obs_queue.push(preprocess(obs, sess))
        state = collect_frames(obs_queue, nframes=agent_history)
        # update episode totals
        episode_return += reward
        episode_steps += 1
        # update global step count
        global_step += 1
        if episode_steps % agent.update_freq == 0 or done:
            # update networks
            targets = agent.targets(state, rewards, done, sess)
            summary = agent.update(states, actions, targets, sess)
            writer.add_summary(summary, global_step)
            # reset arrays
            states.clear()
            actions.clear()
            rewards.clear()
        if done:
            break
    episode_fps = (global_step - start_step) / (time.time() - start_time)
    return episode_return, episode_steps, episode_fps, global_step

def log_episode(writer, episode_return, episode_steps, episode_fps, global_step):
    log_scalar(writer, 'return', episode_return, global_step)
    log_scalar(writer, 'steps', episode_steps, global_step)
    log_scalar(writer, 'fps', episode_fps, global_step)

def log_scalar(writer, tag, value, step):
    value = [tf.Summary.Value(tag=tag, simple_value=value)]
    summary = tf.Summary(value=value)
    writer.add_summary(summary, step)

def print_items(items):
    print(', '.join([f"{k}: {v}" for (k,v) in items.items()]))

def train(env_name='CartPole-v0',
          device='/cpu:0',
          hidden_units=[64],
          learning_rate=1e-3,
          beta=0.0,
          discount_factor=0.99,
          update_freq=5,
          max_episodes=1000,
          pass_condition=195.0,
          log_freq=25,
          ckpt_freq=25,
          base_dir=None,
          render=True,
          atari=False):
    """
        Train an a2c agent on `env_name`.
    """

    print("Creating graph...")
    env = gym.make(env_name)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    with tf.device(device):
        if atari:
            states_pl = tf.placeholder(tf.float32, [None, 84, 84, agent_history])
        else:
            states_pl = tf.placeholder(tf.float32, [None, state_dim])
        actions_pl = tf.placeholder(tf.int32, [None])
        targets_pl = tf.placeholder(tf.float32, [None])
        if atari:
            values = cnn(
                tf.cast(states_pl, tf.float32) / 255.0,
                1,
                scope='value'
            )
            policy_logits = cnn(
                tf.cast(states_pl, tf.float32) / 255.0,
                action_dim,
                scope='value'
            )
        else:
            values = mlp(states_pl, hidden_units + [1], tf.tanh)
            policy_logits = mlp(states_pl, hidden_units + [action_dim], tf.tanh)

    print("Creating agent...")
    placeholders = {
        'states': states_pl,
        'actions': actions_pl,
        'targets': targets_pl
    }
    networks = {
        'value': values,
        'policy': policy_logits
    }
    agent = ActorCritic(
        placeholders,
        networks,
        lr=learning_rate,
        beta=beta,
        update_freq=update_freq,
        gamma=discount_factor,
        device=device,
    )

    print("Setting up directories...")
    if base_dir is not None:
        ckpt_dir, log_dir, meta_dir = create_directories(env_name, "a2c", base_dir)
        meta = {
            'env_name': env_name,
        }
        with open(meta_dir + '/meta.json', 'w') as file:
            json.dump(meta, file, indent=2)
    else:
        ckpt_dir = log_dir = None

    print("Starting training...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        now = datetime.today()
        date_string = now.strftime("%Y-%m-%d-%H:%M:%S.%f")
        writer = tf.summary.FileWriter(log_dir + '/' + date_string, sess.graph)
        returns = []
        global_step = 0
        for episode in range(max_episodes):
            data = run_episode(env, agent, sess, writer, global_step)
            episode_return, episode_steps, episode_fps, global_step = data
            returns += [episode_return]
            avg_return = sum(returns[-100:]) / 100
            if episode % log_freq == 0 or avg_return > pass_condition:
                print_items(
                    {
                        'episode': episode,
                        'return': episode_return,
                        'average': avg_return,
                        'step': global_step
                    }
                )
                log_episode(
                    writer,
                    episode_return,
                    episode_steps,
                    episode_fps,
                    global_step
                )
            if (episode + 1) % ckpt_freq == 0:
                if ckpt_dir is not None:
                    agent.save(ckpt_dir + "/ckpt", global_step, sess)
            if avg_return > pass_condition:
                agent.save(ckpt_dir + "/ckpt", global_step, sess)
                print("passed!")
                break


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('env_name', 'CartPole-v0', """Gym environment.""")
tf.app.flags.DEFINE_string('device', '/cpu:0', """'/cpu:0' or '/gpu:0'.""")
tf.app.flags.DEFINE_string('hidden_units', '64', """Size of hidden layers.""")
tf.app.flags.DEFINE_float('learning_rate', 0.00025, """Initial learning rate.""")
tf.app.flags.DEFINE_float('beta', 1., """Entropy penalty strength.""")
tf.app.flags.DEFINE_float('discount_factor', 0.99, """Discount factor in update.""")
tf.app.flags.DEFINE_integer('update_freq', 5, """Number of actions between updates.""")
tf.app.flags.DEFINE_integer('max_episodes', 10000, """Episodes per train/test run.""")
tf.app.flags.DEFINE_float('pass_condition', 195., """Average score considered passing environment.""")
tf.app.flags.DEFINE_integer('ckpt_freq', 25, """Episodes per checkpoint.""")
tf.app.flags.DEFINE_integer('log_freq', 25, """Steps per log.""")
tf.app.flags.DEFINE_string('base_dir', '.', """Base directory for checkpoints and logs.""")
tf.app.flags.DEFINE_boolean('render', True, """Render episodes (once per `ckpt_freq` in training mode).""")
tf.app.flags.DEFINE_boolean('atari', False, """Is it an Atari environment?""")

if __name__ == "__main__":
    train(env_name=FLAGS.env_name,
          device=FLAGS.device,
          hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],
          learning_rate=FLAGS.learning_rate,
          beta=FLAGS.beta,
          discount_factor=FLAGS.discount_factor,
          update_freq=FLAGS.update_freq,
          max_episodes=FLAGS.max_episodes,
          pass_condition=FLAGS.pass_condition,
          log_freq=FLAGS.log_freq,
          ckpt_freq=FLAGS.log_freq,
          base_dir=FLAGS.base_dir,
          render=FLAGS.render,
          atari=FLAGS.atari)
