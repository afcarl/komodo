import gym
import numpy as np
import tensorflow as tf
import time
from komodo.keras import cnn

class DQN():

    def __init__(
        self,
        lr=0.00025,  # learning rate
        gamma=0.99,  # discount factor
        init_epsilon=1.0,  # initial exploration rate
        min_epsilon=0.05,  # minimum exploration rate
        anneal_steps=1000000,  # exploration annealing steps
        clone_steps=10000,  # steps between cloning network
        update_freq=4,  # actions between updates
        agent_history=4,  # observations per state
        device='/gpu:0',  # device for *primary* graph
        atari=True):

        self.lr = lr
        self.epislon = epsilon
        self.gamma = gamma
        self.device = device
        self.atari = atari

        # construct graph
        with tf.device(device):

            # placeholders
            states_pl = tf.placeholder(tf.uint8, [None, 84, 84, agent_history])
            actions_pl = tf.placeholder(tf.int32, [None])
            targets_pl = tf.placeholder(tf.float32, [None])

            # value networks
            action_values = cnn(
                tf.cast(states_pl, tf.float32) / 255.0,
                n_actions,
                scope='value')
            target_values = cnn(
                tf.cast(states_pl, tf.float32) / 255.0,
                n_actions,
                scope='target')

            # action selection
            greedy_action = tf.arg_max(action_values, dimension=1)
            target_actions = tf.arg_max(target_values, dimension=1)

            # training op
            loss = tf.losses.mean_squared_error(targets_pl, values) # *no* error clipping
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

            # cloning op
            source = tf.get_default_graph().get_collection('trainable_variables', scope='value')
            target = tf.get_default_graph().get_collection('trainable_variables', scope='target')
            clone_ops = [tf.assign(t, s, name='clone') for t,s in zip(target, source)]

        # preprocessing ops
        # frame_pl = tf.placeholder(tf.uint8, [210, 160, 3])
        # preprocess_op = preprocess_tf(frame_pl)  # uint8, 84 x 84

        # tensorboard
        imgs = tf.reshape(states_pl[0, :, :, :], [-1, 84, 84, agent_history])
        tf.summary.image('state', imgs, max_outputs=4)
        tf.summary.histogram('action_values', values)
        tf.summary.scalar('loss', loss)
        summary_op = tf.summary.merge_all()

        # checkpoints
        saver = tf.train.Saver()

        # handles
        self.states_pl = states_pl
        self.actions_pl = actions_pl
        self.targets_pl = targets_pl
        self.action_values = action_values
        self.target_values = target_values
        self.greedy_action = greedy_action
        self.target_actions = target_actions
        self.train_op = train_op
        self.clone_ops = clone_ops
        # self.preprocess_op = preprocess_op
        self.summary_op = summary_op
        self.saver = saver

    def select_action(self, state, sess):
        if np.random.random() < self.epsilon:
            action = np.random.randint(n_actions)
        else:
            action = sess.run(self.greedy_action,
                feed_dict={
                    states_pl: state.reshape(1, 84, 84, self.agent_history)
                })
            action = action[0]

    # def preprocess(self, frame, sess):
    #     return sess.run(self.preprocess_op, {self.frame_pl: frame})

    def clone(self, sess):
        """Clone `action_values` network to `target_values`."""
        sess.run(self.clone_ops)

    def update(self, states, actions, rewards, next_states, done_flags):

        # compute targets
        targets = sess.run(
            self.targets,
            feed_dict={
                self.states_pl: next_states,
            })
        targets = rewards + ~done_flags * self.gamma * targets

        # perform update
        summary, _ = sess.run([self.summary_op, self.train_op],
            feed_dict={
                self.states_pl: states,
                self.actions_pl: actions,
                self.targets_pl: targets,
            })
        return summary

    def anneal_epsilon(self, step):
        """Linear annealing of `init_epsilon` to `min_epsilon` over `anneal_steps`."""
        rate = (self.init_epsilon - self.min_epsilon) / self.anneal_steps
        self.epsilon = self.init_epsilon - step * rate
        return self.epsilon

    def save(self, path, step, sess):
        self.saver.save(sess, save_path=path, global_step=step)

class ReplayMemory():

    def __init__(
        self,
        env,
        min_memory_size=10000,
        max_memory_size=10000):

        assert type(env) == AtariEnvironment
        print('Filling replay memory...')
        start = time.time()
        queue = Queue(size=max_memory_size)
        while len(queue) < min_memory_size:
            state = env.reset()
            while True:
                action = env.random_action()
                next_state, reward, done, info = env.step(action)
                queue.push([state, action, reward, next_state, done])
                state = next_state.copy()
                if done or len(queue) == min_memory_size:
                    break
        elapsed_time = time.time() - start
        self.queue = queue
        print(f"done (elapsed time: {elapsed_time:.2f})")

    def sample(self, size):
        """Sample `size` transitions from `memory` uniformly"""
        idx = np.random.choice(range(len(self.queue)), size)
        batch = [self.queue[i] for i in idx]
        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.clip(np.array([b[2] for b in batch]), -1, 1)  # reward clipping
        next_states = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])
        return states, actions, rewards, next_states, dones

    def append(self, transition):
        """Push a new [state, action, reward, next_state, done_flag] observation onto memory replay."""
        self.queue.push(transition)
