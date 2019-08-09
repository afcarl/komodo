import gym
import numpy as np
import tensorflow as tf
from versterken.keras import mlp
from versterken.utils import dimensions, bootstrapped_values

# class AtariProximalPolicy():
#
#     def __init__(self, lr, beta, gamma, epsilon, device='/gpu:0'):
#
#         with tf.device(device):
#
#             # placeholders
#             states_pl = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states')
#             actions_pl = tf.placeholder(tf.int32, [None], name='actions')
#             targets_pl = tf.placeholder(tf.float32, [None], name='targets')
#
#             # networks
#             values, policy_logits = cnn(states_pl / 255.0, 6, shared=True)
#             action_mask = tf.one_hot(actions_pl, int(policy_logits.shape[1]))
#             policy = tf.reduce_sum(action_mask * tf.nn.log_softmax(policy_logits), axis=1)
#
#             # advantage
#             advantages = targets_pl - tf.stop_gradient(values)
#
#             # actions
#             action_sample = tf.multinomial(logits=policy_logits, num_samples=1)
#             action_sample = tf.squeeze(action_sample, axis=1)
#
#             # losses
#             ratio = policy / tf.stop_gradient(policy)
#             clipped = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
#             policy_loss = tf.minimum(ratio * advantages, clipped * advantages)
#             value_loss = tf.reduce_mean(tf.square(targets_pl - values))
#             entropy_loss = entropy_beta * tf.reduce_mean(
#                 tf.multiply(
#                     tf.nn.softmax(policy_logits),  # probabilities
#                     tf.nn.log_softmax(policy_logits)  # log probabilities
#                 )
#             )
#             loss = policy_loss + entropy_loss + value_loss
#
#             # update
#             optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-3)
#             train_op = optimizer.minimize(loss)
#
#         # tensorboard
#         tf.summary.scalar('policy_loss', policy_loss)
#         tf.summary.scalar('entropy_loss', entropy_loss)
#         tf.summary.scalar('value_loss', value_loss)
#         tf.summary.scalar('loss', loss)
#         tf.summary.histogram('policy_logits', policy_logits)
#         tf.summary.histogram('values', values)
#         tf.summary.histogram('advantages', advantages)
#         summary_op = tf.summary.merge_all()
#
#         # checkpoints
#         saver = tf.train.Saver()
#
#         # define handles
#         self.states_pl = states_pl
#         self.actions_pl = actions_pl
#         self.targets_pl = targets_pl
#         self.values = values
#         self.policy_logits = policy_logits
#         self.advantages = advantages
#         self.action_sample = action_sample
#         self.train_op = train_op
#         self.summary_op = summary_op
#         self.saver = saver
#
#     def select_actions(self, states, sess):
#         feed_dict = {self.states_pl: states}
#         return sess.run(self.action_sample, feed_dict=feed_dict)  # NOTE: return **list**
#
#     def compute_targets(self, states, rewards, flags, sess):
#         cumulative_rewards = accumulate(rewards, self.gamma)
#         feed_dict = {self.states_pl: states}
#         terminal_values = flags * sess.run(self.values, feed_dict=feed_dict)
#         # TODO
#         targets = ....
#
#         return targets
#
#     def update(self, states, actions, targets, sess, logging=False):
#
#         # perform update
#         feed_dict={
#             self.states_pl: states,
#             self.actions_pl: actions,
#             self.targets_pl: targets,
#         }
#         if logging:
#             summary, _ = sess.run(
#                 [self.summary_op, self.train_op,],
#                 feed_dict=feed_dict)
#             return summary
#         else:
#             sess.run(self.train_op, feed_dict=feed_dict)
#             return None

class ProximalPolicy():

    def __init__(
        self,
        lr=0.001,
        entropy_beta=0.01,
        value_beta=0.5,
        gamma=0.99,
        epsilon=0.1,
        device='/gpu:0'):

        self.gamma = gamma

        with tf.device(device):

            # placeholders
            states_pl = tf.placeholder(tf.float32, [None, 4], name='states')
            actions_pl = tf.placeholder(tf.int32, [None], name='actions')
            targets_pl = tf.placeholder(tf.float32, [None], name='targets')

            # networks
            values = mlp(states_pl, [64, 32, 1], scope='value')
            policy_logits = mlp(states_pl, [64, 32, 2], scope='policy')
            action_mask = tf.one_hot(actions_pl, int(policy_logits.shape[1]))
            policy = tf.reduce_sum(action_mask * tf.nn.softmax(policy_logits), axis=1)

            # advantage
            advantages = targets_pl - tf.stop_gradient(values)

            # actions
            action_sample = tf.multinomial(logits=policy_logits, num_samples=1)
            action_sample = tf.squeeze(action_sample, axis=1)

            # losses
            ratio = policy / tf.stop_gradient(policy)
            clipped = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
            policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
            value_loss = tf.reduce_mean(tf.square(targets_pl - values))
            entropy_loss = tf.reduce_mean(
                tf.multiply(
                    tf.nn.softmax(policy_logits),  # probabilities
                    tf.nn.log_softmax(policy_logits)  # log probabilities
                )
            )
            loss = policy_loss + entropy_beta * entropy_loss + value_beta * value_loss

            # update
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-3)
            train_op = optimizer.minimize(loss)

        # tensorboard
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('entropy_loss', entropy_loss)
        tf.summary.scalar('value_loss', value_loss)
        tf.summary.scalar('loss', loss)
        tf.summary.histogram('policy_logits', policy_logits)
        tf.summary.histogram('values', values)
        tf.summary.histogram('advantages', advantages)
        summary_op = tf.summary.merge_all()

        # checkpoints
        saver = tf.train.Saver()

        # define handles
        self.state_dim = dimensions(states_pl)
        self.action_dim = dimensions(actions_pl)
        self.states_pl = states_pl
        self.actions_pl = actions_pl
        self.targets_pl = targets_pl
        self.values = values
        self.policy_logits = policy_logits
        self.advantages = advantages
        self.action_sample = action_sample
        self.train_op = train_op
        self.summary_op = summary_op
        self.saver = saver

    def select_actions(self, states, sess):
        feed_dict = {self.states_pl: states.reshape(self.state_dim)}
        return sess.run(self.action_sample, feed_dict=feed_dict)  # NOTE: returns **list**

    def get_values(self, states, sess):
        feed_dict = {self.states_pl: states.reshape(self.state_dim)}
        return sess.run(self.values, feed_dict=feed_dict)  # NOTE: returns **list**

    def bundle(self, data, info):
        """Bundle batch data and info into format for update."""
        states = []
        actions = []
        targets = []
        returns = []
        steps = []
        for (bd, bi) in zip(data, info):
            s, a, r, tv, d = bd
            states += s
            actions += a
            targets += bootstrapped_values(tv, r, self.gamma)
            _, er, _ = bi
            if er is not None:
                returns += [er]
        episodes = len(returns)
        steps = len(states)
        return states, actions, targets, episodes, returns, steps

    def update(self, states, actions, targets, sess, logging=True):

        # perform update
        feed_dict={
            self.states_pl: states,
            self.actions_pl: actions,
            self.targets_pl: targets,
        }
        if logging:
            summary, _ = sess.run(
                [self.summary_op, self.train_op,],
                feed_dict=feed_dict)
            return summary
        else:
            sess.run(self.train_op, feed_dict=feed_dict)
            return None

    def save(self, path, step, sess):
        self.saver.save(sess, save_path=path, global_step=step)
