import os
from datetime import datetime
import tensorflow as tf

def create_directories(env_name, agent_name, base_dir = "."):
    """Create a directories to save logs and checkpoints.

        E.g. ./checkpoints/LunarLander/dqn-vanilla/5-14-2019-9:37:25.43/

        # Arguments
        - env_name (string): name of Gym environment.
        - agent_name (string): name of learning algorithm.
    """

    now = datetime.today()
    date_string = now.strftime("%Y-%m-%d-%H:%M:%S.%f")
    ckpt_dir = "/".join([base_dir, "checkpoints", env_name, agent_name, date_string])
    log_dir = "/".join([base_dir, "logs", env_name, agent_name, date_string])
    meta_dir = "/".join([base_dir, "meta", env_name, agent_name, date_string])
    os.makedirs(ckpt_dir)
    os.makedirs(log_dir)
    os.makedirs(meta_dir)
    return ckpt_dir, log_dir, meta_dir

def log_scalar(writer, tag, value, step):
    value = [tf.Summary.Value(tag=tag, simple_value=value)]
    summary = tf.Summary(value=value)
    writer.add_summary(summary, step)

def log_episode(writer, episode_return, episode_steps, episode_fps, global_step):
    log_scalar(writer, 'return', episode_return, global_step)
    log_scalar(writer, 'steps', episode_steps, global_step)
    log_scalar(writer, 'fps', episode_fps, global_step)

def print_items(items):
    print(', '.join([f"{k}: {v}" for (k,v) in items.items()]))

def find_latest_checkpoint(load_path, prefix):
    """Find the latest checkpoint in dir at `load_path` with prefix `prefix`

        E.g. ./checkpoints/dqn-vanilla-CartPole-v0-GLOBAL_STEP would use find_latest_checkpoint('./checkpoints/', 'dqn-vanilla-CartPole-v0')
    """
    files = os.listdir(load_path)
    matches = [f for f in files if f.find(prefix) == 0]  # files starting with prefix
    max_steps = np.max(np.unique([int(m.strip(prefix).split('.')[0]) for m in matches]))
    latest_checkpoint = load_path + prefix + '-' + str(max_steps)
    return latest_checkpoint

def dimensions(tensor):
    """Return dimensions of a Tensor in a format compatible with numpy.

        TensorFlow allows `None` dimensions, but numpy does not.
    """
    dims = tensor.get_shape().as_list()
    assert dims.count(None) < 2, "numpy dimensions can contain at most one -1"
    for i,d in enumerate(dims):
        if d is None:
            dims[i] = -1
    return dims

class SimpleBatchGenerator():

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.state = self.env.reset()
        self.episode_return = 0.0
        self.episode_steps = 0

    def sample(self, n, sess):
        """Sample a n-step trajectory from the environment."""

        # batch trajectory data
        states = []
        actions = []
        rewards = []

        # batch episode data
        episode_return = None
        episode_steps = None
        batch_steps = 0  # numer of steps this batch

        # generate trajectories
        for step in range(n):

            # perform an action
            action = self.agent.select_actions(self.state, sess)[0]
            next_state, reward, done, info = self.env.step(action)

            # update batch info
            batch_steps += 1

            # update local environment data
            states += [self.state.copy()]
            actions += [action]
            rewards += [reward]

            # update global state
            self.state = next_state.copy()
            self.episode_return += reward
            self.episode_steps += 1

            # check if we reached end of episode
            if done:
                terminal_value = 0.0
                episode_return = self.episode_return
                episode_steps = self.episode_steps
                # reset environment
                self.state = self.env.reset()
                self.episode_return = 0.0
                self.episode_steps = 0
                break
            else:
                terminal_value = self.agent.get_values(self.state, sess)
                episode_return = None
                episode_steps = None

        batch_data = states, actions, rewards, terminal_value, done
        batch_info = batch_steps, episode_return, episode_steps
        return batch_data, batch_info

class BatchGenerator():

    def __init__(self, envs, agent):
        self.generators = [SimpleBatchGenerator(env, agent) for env in envs]

    def sample(self, n, sess):
        """Sample a n-step trajectory from each environment."""

        batch_data = []
        batch_info = []
        for g in self.generators:
            bd, bi = g.sample(n, sess)
            batch_data += [bd]
            batch_info += [bi]
        return batch_data, batch_info

class NstepBatchGenerator():

    def __init__(self, envs, agent):
        self.envs = envs
        self.agent = agent
        self.size = len(envs)
        self.states = [e.reset() for e in self.envs]
        self.episode_rewards = [0.0] * self.size
        self.episode_steps = [0] * self.size

    def sample(self, n, sess):
        """Sample a n-step trajectory from each environment."""

        # batch trajectory data
        init_states = self.states.copy()
        states = self.states.copy()  # will keep track of terminal states...
        discounted_rewards = [0.0] * self.size
        done_flags = [False] * self.size  # True if environment reaches end of epsidoe this sample

        # batch episode data
        episode_rewards = []
        episode_steps = []
        batch_steps = 0  # total steps performed across all environments

        # generate trajectories
        for step in range(n):
            actions = self.agent.select_actions(states, sess)  # *simultaneous* action selection!
            if step == 0:
                init_actions = actions.copy()
            # perform actions env-by-env...
            for idx in range(self.size):
                if not done_flags[idx]:  # check if environment already reached end of episode this sample
                    action = actions[idx]
                    state, reward, done, info = self.envs[idx].step(action)

                    # update batch info
                    batch_steps += 1

                    # update local environment data
                    states[idx] = state
                    discounted_rewards[idx] += self.agent.gamma ** step * reward

                    # update global environment data
                    self.states[idx] = state
                    self.episode_rewards[idx] += reward
                    self.episode_steps[idx] += 1

                    # check if we reached end of episode
                    if done:
                        done_flags[idx] = True
                        episode_rewards += [self.episode_rewards[idx]]
                        episode_steps += [self.episode_steps[idx]]
                        # reset environment
                        self.states[idx] = self.envs[idx].reset()
                        self.episode_rewards[idx] = 0.0
                        self.episode_steps[idx] = 0
                else:
                    pass

        batch_data = (init_states, init_actions, discounted_rewards, states, done_flags)
        batch_info = batch_steps, len(episode_rewards), episode_rewards, episode_steps
        return batch_data, batch_info

def bootstrapped_values(terminal_value, rewards, gamma):
    """Calculate targets used to update policy and value functions."""
    targets = []
    R = terminal_value
    for r in rewards[-1::-1]:
        R = r + gamma * R
        targets += [R]
    return targets[-1::-1]  # reverse to match original ordering

def bundle(batch_data, batch_info, gamma):
    """Bundle batch data and info into format for update."""
    states = []
    actions = []
    targets = []
    returns = []
    steps = []
    for (bd, bi) in zip(batch_data, batch_info):
        s, a, r, tv, d = bd
        states += s
        actions += a
        targets += bootstrapped_values(tv, r, gamma)
        _, er, es = bi
        if er is not None:
            returns += [er]
        if es is not None:
            steps += [es]
    return states, actions, targets, returns, steps
