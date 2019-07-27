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
from versterken.atari import collect_frames
from versterken.queue import Queue
from versterken.utils import create_directories, log_scalar, log_episode, print_items
from versterken.keras import mlp, cnn

# TODO: gradient clipping?
# TODO: multiple environments?
# TODO: learning_rate annealing

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
    episode_return = 0
    episode_steps = 0
    start_step = global_step
    states = []
    actions = []
    rewards = []
    obs = env.reset()
    obs_queue = Queue(init_values=[agent.preprocess(obs, sess)], size=agent.history)
    state = collect_frames(obs_queue, nframes=agent.history)
    if render == True:
        env.render()
    start_time = time.time()
    while True:
        # perform an action
        action = agent.action(state.reshape(1, 84, 84, agent.history), sess)
        obs, reward, done, info = env.step(action)
        if render == True:
            env.render()
        # record transition
        states += [state]
        actions += [action]
        rewards += [reward]
        # calculate new state
        obs_queue.push(agent.preprocess(obs, sess))
        state = collect_frames(obs_queue, nframes=agent.history)
        # update episode totals
        episode_return += reward
        episode_steps += 1
        # update global step count
        global_step += 1
        if episode_steps % agent.update_freq == 0 or done:
            # update networks
            targets = agent.targets(state.reshape(1, 84, 84, agent.history), rewards, done, sess)
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

def train(env_name='CartPole-v0',
          device='/cpu:0',
          hidden_units=[64],
          learning_rate=1e-3,
          beta=0.0,
          discount_factor=1.0,
          update_freq=4,
          agent_history=1,
          max_episodes=1000,
          pass_condition=195.0,
          log_freq=5,
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
                scope='policy'
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
        history=agent_history,
        device=device,
        atari=atari,
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
            if (episode + 1) % ckpt_freq == 0:  # render (if render == True)
                if atari:
                    data = run_atari(env, agent, sess, writer, global_step, render)
                else:
                    data = run_episode(env, agent, sess, writer, global_step, render)
            else:  # no rendering
                if atari:
                    data = run_atari(env, agent, sess, writer, global_step)
                else:
                    data = run_episode(env, agent, sess, writer, global_step)
            episode_return, episode_steps, episode_fps, global_step = data
            returns += [episode_return]
            avg_return = sum(returns[-100:]) / min(len(returns), 100)
            if episode % log_freq == 0 or avg_return > pass_condition:
                print_items(
                    {
                        'episode': episode,
                        'return': episode_return,
                        'average': avg_return,
                        'step': global_step,
                        'fps': episode_fps
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
tf.app.flags.DEFINE_string('env_name', 'Pong-v4', """Gym environment.""")
tf.app.flags.DEFINE_string('device', '/gpu:0', """'/cpu:0' or '/gpu:0'.""")
tf.app.flags.DEFINE_string('hidden_units', '64', """Size of hidden layers.""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, """Initial learning rate.""")
tf.app.flags.DEFINE_float('beta', 0.01, """Entropy penalty strength.""")
tf.app.flags.DEFINE_float('discount_factor', 0.99, """Discount factor in update.""")
tf.app.flags.DEFINE_integer('update_freq', 8, """Number of actions between updates.""")
tf.app.flags.DEFINE_integer('agent_history', 4, """Number of actions between updates.""")
tf.app.flags.DEFINE_integer('max_episodes', 10000, """Episodes per train/test run.""")
tf.app.flags.DEFINE_float('pass_condition', 19.5, """Average score considered passing environment.""")
tf.app.flags.DEFINE_integer('ckpt_freq', 10, """Episodes per checkpoint.""")
tf.app.flags.DEFINE_integer('log_freq', 1, """Steps per log.""")
tf.app.flags.DEFINE_string('base_dir', '.', """Base directory for checkpoints and logs.""")
tf.app.flags.DEFINE_boolean('render', True, """Render episodes (once per `ckpt_freq` in training mode).""")
tf.app.flags.DEFINE_boolean('atari', True, """Is it an Atari environment?""")

if __name__ == "__main__":
    train(env_name=FLAGS.env_name,
          device=FLAGS.device,
          hidden_units=[int(i) for i in FLAGS.hidden_units.split(',')],  # ignored if atari == False
          learning_rate=FLAGS.learning_rate,
          beta=FLAGS.beta,
          discount_factor=FLAGS.discount_factor,
          update_freq=FLAGS.update_freq,
          agent_history=FLAGS.agent_history,
          max_episodes=FLAGS.max_episodes,
          pass_condition=FLAGS.pass_condition,
          log_freq=FLAGS.log_freq,
          ckpt_freq=FLAGS.ckpt_freq,
          base_dir=FLAGS.base_dir,
          render=FLAGS.render,
          atari=FLAGS.atari)
