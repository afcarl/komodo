import numpy as np
from versterken.atari import AtariEnvironment
from versterken.dqn import DQN, ReplayMemory
from versterken.utils import create_directories, log_scalar

def train(id, batch_size, base_dir='./examples'):
    """
        - `id`: id of the environment (e.g., "Pong-v4")
        - `batch_size`: transitions per batch
    """

    _, log_dir, _ = create_directories('Pong-v4', "a2c", base_dir)
    env = AtariEnvironment(gym.make(id))
    agent = DQN()
    memory = ReplayMemory(env)
    global_returns = []
    global_steps = 0
    global_start = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        agent.clone(sess)
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        while True:
            # begin episode
            state = env.reset()
            episode_return = 0.0
            episode_steps = 0
            start = time.time()
            while True:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                transition = [state, action, reward, next_state, done]
                current_memories = memory.append(transition)
                state = next_state.copy()
                episode_return += reward
                episode_steps += 1
                global_steps += 1
                epsilon = agent.anneal_epsilon(global_steps)
                if episode_steps % agent.update_freq == 0:
                    batch = memory.sample(batch_size)
                    states, actions, rewards, next_states, done_flags = batch
                    summary = agent.update(
                        states,
                        actions,
                        rewards,
                        next_states,
                        done_flags,
                        sess
                    )
                if global_steps % agent.clone_steps == 0:
                    agent.clone(sess)
                if done:
                    break
            episode_fps = episode_steps / (time.time() - start)
            # end episode
            global_episodes += 1
            global_returns += [episode_return]
            global_time = time.time() - global_start
            avg_return = np.mean(global_returns[-100:])
            log_scalar(writer, 'avg_return', avg_return, global_steps)
            log_scalar(writer, 'return', episode_return, global_steps)
            log_scalar(writer, 'steps', episode_steps, global_steps)
            log_scalar(writer, 'fps', episode_fps, global_steps)
            log_scalar(writer, 'epsilon', epsilon, global_steps)
            writer.add_summary(summary, global_steps)
            print(f"step={global_steps}, episode={global_episodes}, avg_return={avg_return:.2f}, elapsed={global_time:.2f}, fps={episode_fps:.2f}, epsilon={epsilon:.2f}")
            if avg_return > pass_condition:
                print(f"Pass condition reached! (avg_return={avg_return:.2f})")
                break
