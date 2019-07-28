import gym
import numpy as np
import tensorflow as tf
from versterken.atari import rgb_to_grayscale
from versterken.keras import clip_by_norm

class ActorCritic():

    def __init__(
            self,
            placeholders,
            networks,
            lr=0.0025,
            beta=0.01,
            update_freq=4,
            gamma=1.0,
            history=1,
            device='/cpu:0',
            atari=False
        ):

        # construct graph
        with tf.device(device):
            # placeholders
            states_pl = placeholders['states']
            actions_pl = placeholders['actions']
            targets_pl = placeholders['targets']

            # networks
            values = networks['value']
            policy_logits = networks['policy']
            action_mask = tf.one_hot(actions_pl, int(policy_logits.shape[1]))
            policy = tf.reduce_sum(action_mask * tf.nn.log_softmax(policy_logits), axis=1)  # π(a | s)

            # losses
            policy_loss = -tf.reduce_mean(policy * (targets_pl - tf.stop_gradient(values)))
            value_loss = tf.reduce_mean(tf.square(targets_pl - values))
            entropy_loss = beta * tf.reduce_mean(
                tf.multiply(
                    tf.nn.softmax(policy_logits),  # probabilities
                    tf.nn.log_softmax(policy_logits)  # log probabilities
                )
            )

            # updates
            # value_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-3)
            value_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
            value_gradients = value_optimizer.compute_gradients(value_loss)
            value_gradients = clip_by_norm(value_gradients, clip_norm=0.1)
            value_update = value_optimizer.apply_gradients(value_gradients)

            # policy_optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-3)
            policy_optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
            policy_gradients = policy_optimizer.compute_gradients(policy_loss)
            policy_gradients = clip_by_norm(policy_gradients, clip_norm=0.1)
            policy_update = policy_optimizer.apply_gradients(policy_gradients)

            # value_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(value_loss)
            # policy_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(policy_loss + entropy_loss)

            # action selection
            action_sample = tf.squeeze(tf.multinomial(logits=policy_logits, num_samples=1), axis=1)

        # tensorboard
        tf.summary.histogram('values', values)
        tf.summary.histogram('policy_logits', policy_logits)
        tf.summary.histogram('value_loss', value_loss)
        tf.summary.histogram('policy_loss', policy_loss)
        tf.summary.histogram('entropy_loss', entropy_loss)
        summary_op = tf.summary.merge_all()

        # checkpoints
        saver = tf.train.Saver()

        # define handles
        self.states_pl = states_pl
        self.actions_pl = actions_pl
        self.targets_pl = targets_pl
        self.values = values
        self.policy_update = policy_update
        self.value_update = value_update
        self.action_sample = action_sample
        self.summary_op = summary_op
        self.saver = saver
        self.update_freq = update_freq
        self.gamma = gamma
        self.history = history
        self.atari = atari

        # preprocessor
        self.frame_pl = tf.placeholder(tf.uint8, [210, 160, 3])
        self.preprocess_op = rgb_to_grayscale(self.frame_pl)  # uint8, 84 x 84

    def action(self, state, sess):
        if self.atari:
            feed_dict = {self.states_pl: state.reshape(1, 84, 84, agent.history)}
        else:
            feed_dict = {self.states_pl: state.reshape(1, -1)}
        return sess.run(self.action_sample, feed_dict=feed_dict)[0]

    def targets(self, terminal_states, rewards, flags, sess):

        n = len(terminal_states)

        if self.atari:
            feed_dict = {
                self.states_pl: np.reshape(terminal_states, (n, 84, 84, agent.history))
            }
        else:
            feed_dict = {
                self.states_pl: np.reshape(terminal_states, (n, -1))
            }

        terminal_values = sess.run(
            self.values,
            feed_dict={
                self.states_pl: terminal_states
            }
        )
        # print(terminal_values)
        # print(flags)
        terminal_values *= ~np.array(flags)
        # print(terminal_values)
        targets = [bootstrapped_values(tv, r, self.gamma) for tv, r in zip(terminal_values, rewards)]
        return np.concatenate(targets)

    def preprocess(self, frame, sess):
        return sess.run(self.preprocess_op, {self.frame_pl: frame})

    def update(self, states, actions, targets, sess):
        summary, _, _ = sess.run(
            [
                self.summary_op,
                self.policy_update,
                self.value_update
            ],
            feed_dict={
                self.states_pl: states,
                self.actions_pl: actions,
                self.targets_pl: targets
            }
        )
        return summary

    def save(self, path, step, sess):
        self.saver.save(sess, save_path=path, global_step=step)

def bootstrapped_values(terminal_value, rewards, gamma):
    """Calculate targets used to update policy and value functions."""
    targets = []
    R = terminal_value
    for r in rewards[-1::-1]:
        R = r + gamma * R
        targets += [R]
    return targets[-1::-1]  # reverse to match original ordering


class Generator():

    def __init__(self, id, agent):
        self.env = gym.make(id)
        self.agent = agent
        self.states = []
        self.actions = []
        self.rewards = []
        self.state = self.env.reset()
        self.total = 0
        self.steps = 0

    def sample(self, n, sess):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        for i in range(n):
            state = self.state
            if self.agent is not None:
                # print("agent.action called!")
                action = self.agent.action(self.state, sess)
            else:
                action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            self.states += [state]
            self.actions += [action]
            self.rewards += [reward]
            self.state = next_state
            self.total += reward
            self.steps += 1
            info = (self.total, self.steps)
            if done:
                self.state = self.env.reset()
                self.total = 0
                self.steps = 0
                break
        return self.states, self.actions, self.rewards, next_state, done, info
