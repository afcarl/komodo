import gym
import numpy as np
import tensorflow as tf
from komodo.keras import clip_by_norm, mlp
from komodo.utils import dimensions

class AtariActorCritic():

    def __init__(
            self,
            # placeholders,
            # networks,
            lr=7e-4,
            entropy_beta=0.01,
            value_beta=0.5,
            gamma=0.99,
            tmax=5,
            max_grad_norm=0.5,
            device='/gpu:0',
            atari=True):

        self.gamma = gamma
        self.tmax = tmax
        self.atari = atari

        # construct graph
        with tf.device(device):

            # placeholders
            states_pl = tf.placeholder(tf.float32, [None, 84, 84, 4], name='states')
            actions_pl = tf.placeholder(tf.int32, [None], name='actions')
            rewards_pl = tf.placeholder(tf.float32, [None], name='rewards')
            flags_pl = tf.placeholder(tf.float32, [None], name='flags')  # True = episode *not* finished
            targets_pl = tf.placeholder(tf.float32, [None], name='targets')

            # networks
            values, policy_logits = cnn(states_pl / 255.0, 6, shared=True)
            action_mask = tf.one_hot(actions_pl, int(policy_logits.shape[1]))
            policy = tf.reduce_sum(action_mask * tf.nn.log_softmax(policy_logits), axis=1)

            # targets
            targets = rewards_pl + flags_pl * gamma ** tmax * values

            # actions
            action_sample = tf.squeeze(
                tf.multinomial(logits=policy_logits, num_samples=1),
                axis=1
            )

            # losses
            advantage = targets_pl - tf.stop_gradient(values)
            policy_loss = -tf.reduce_mean(policy * advantage)
            value_loss = tf.reduce_mean(tf.square(targets_pl - values))
            entropy_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.multiply(
                        tf.nn.softmax(policy_logits),  # probabilities
                        tf.nn.log_softmax(policy_logits)  # log probabilities
                    ),
                    axis=1
                )
            )
            loss = policy_loss + entropy_beta * entropy_loss + value_beta * value_loss

            # updates
            # optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-5)
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
            # gradients = clip_by_norm(optimizer.compute_gradients(loss), clip_norm=max_grad_norm)
            gradients = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(gradients)

        # construct graph
        # with tf.device(device):
        #     # placeholders
        #     states_pl = placeholders['states']
        #     actions_pl = placeholders['actions']
        #     rewards_pl = placeholders['rewards']
        #     flags_pl = placeholders['flags']
        #     targets_pl = placeholders['targets']
        #
        #     # networks
        #     values = networks['value']
        #     policy_logits = networks['policy']
        #     action_mask = tf.one_hot(actions_pl, int(policy_logits.shape[1]))
        #     policy = tf.reduce_sum(action_mask * tf.nn.log_softmax(policy_logits), axis=1)
        #
        #     # targets
        #     targets = rewards_pl + flags_pl * self.gamma ** self.tmax * values
        #
        #     # actions
        #     action_sample = tf.squeeze(
        #         tf.multinomial(logits=policy_logits, num_samples=1),
        #         axis=1
        #     )
        #
        #     # losses
        #     policy_loss = -tf.reduce_mean(policy * (targets_pl - tf.stop_gradient(values)))
        #     value_loss = 0.5 * tf.reduce_mean(tf.square(targets_pl - values))
        #     entropy_loss = entropy_beta * tf.reduce_mean(
        #         tf.multiply(
        #             tf.nn.softmax(policy_logits),  # probabilities
        #             tf.nn.log_softmax(policy_logits)  # log probabilities
        #         )
        #     )
        #     loss = policy_loss + entropy_loss + value_loss
        #
        #     # updates
        #     optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-3)
        #     gradients = clip_by_norm(optimizer.compute_gradients(loss), clip_norm=max_grad_norm)
        #     train_op = optimizer.apply_gradients(gradients)

        # tensorboard
        tf.summary.scalar('gradient_norm', tf.reduce_mean([tf.norm(g[0]) for g in gradients]))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('entropy_loss', entropy_loss)
        tf.summary.scalar('value_loss', value_loss)
        tf.summary.histogram('policy_logits', policy_logits)
        tf.summary.histogram('values', values)
        tf.summary.histogram('advantage', advantage)
        imgs = tf.transpose(tf.reshape(states_pl[0, :, :, :], [-1, 84, 84, 4]), perm=[3, 1, 2, 0])
        tf.summary.image('state', imgs, max_outputs=4)
        summary_op = tf.summary.merge_all()

        # checkpoints
        saver = tf.train.Saver()

        # define handles
        self.states_pl = states_pl
        self.actions_pl = actions_pl
        self.rewards_pl = rewards_pl
        self.flags_pl = flags_pl
        self.targets_pl = targets_pl
        self.values = values
        self.policy_logits = policy_logits
        self.targets = targets
        self.action_sample = action_sample
        self.train_op = train_op
        self.summary_op = summary_op
        self.saver = saver

    # def action(self, state, sess):
    #     if self.atari:
    #         feed_dict = {self.states_pl: states.reshape(-1, 84, 84, self.tmax)}
    #     else:
    #         feed_dict = {self.states_pl: states.reshape(1, -1)}
    #     return sess.run(self.action_sample, feed_dict=feed_dict)[0]  # NOTE: return **scalar**!

    def select_actions(self, states, sess):
        feed_dict = {self.states_pl: states}
        return sess.run(self.action_sample, feed_dict=feed_dict)  # NOTE: return **list**

    def update(self, states, actions, rewards, next_states, flags, sess, logging=False):

        # calculate targets
        feed_dict = {
            self.rewards_pl: rewards,
            self.states_pl: next_states,
            self.flags_pl: ~np.array(flags)
        }
        targets = sess.run(self.targets, feed_dict=feed_dict)

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

class NstepActorCritic():

    def __init__(
            self,
            lr=0.01,
            entropy_beta=0.01,
            value_beta=0.5,
            gamma=0.99,
            tmax=5,
            max_grad_norm=0.5,
            device='/gpu:0'):

        self.gamma = gamma
        self.tmax = tmax

        # construct graph
        with tf.device(device):

            # placeholders
            states_pl = tf.placeholder(tf.float32, [None, 4], name='states')
            actions_pl = tf.placeholder(tf.int32, [None], name='actions')
            rewards_pl = tf.placeholder(tf.float32, [None], name='rewards')
            flags_pl = tf.placeholder(tf.float32, [None], name='flags')  # True = episode *not* finished
            targets_pl = tf.placeholder(tf.float32, [None], name='targets')

            # networks
            values = mlp(states_pl, [64, 32, 1], scope='value')
            policy_logits = mlp(states_pl, [64, 32, 2], scope='policy')
            action_mask = tf.one_hot(actions_pl, int(policy_logits.shape[1]))
            policy = tf.reduce_sum(action_mask * tf.nn.log_softmax(policy_logits), axis=1)

            # targets
            targets = rewards_pl + flags_pl * gamma ** tmax * values

            # actions
            action_sample = tf.squeeze(
                tf.multinomial(logits=policy_logits, num_samples=1),
                axis=1
            )

            # losses
            advantage = targets_pl - tf.stop_gradient(values)
            policy_loss = -tf.reduce_mean(policy * advantage)
            value_loss = tf.reduce_mean(tf.square(targets_pl - values))
            entropy_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.multiply(
                        tf.nn.softmax(policy_logits),  # probabilities
                        tf.nn.log_softmax(policy_logits)  # log probabilities
                    ),
                    axis=1
                )
            )
            loss = policy_loss + entropy_beta * entropy_loss + value_beta * value_loss

            # updates
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
            gradients = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(gradients)

        # tensorboard
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('entropy_loss', entropy_loss)
        tf.summary.scalar('value_loss', value_loss)
        tf.summary.histogram('policy_logits', policy_logits)
        tf.summary.histogram('values', values)
        tf.summary.histogram('advantage', advantage)
        summary_op = tf.summary.merge_all()

        # checkpoints
        saver = tf.train.Saver()

        # define handles
        self.state_dim = dimensions(states_pl)
        self.action_dim = dimensions(actions_pl)
        self.states_pl = states_pl
        self.actions_pl = actions_pl
        self.rewards_pl = rewards_pl
        self.flags_pl = flags_pl
        self.targets_pl = targets_pl
        self.values = values
        self.policy_logits = policy_logits
        self.targets = targets
        self.action_sample = action_sample
        self.train_op = train_op
        self.summary_op = summary_op
        self.saver = saver

    def select_actions(self, states, sess):
        feed_dict = {self.states_pl: states}
        return sess.run(self.action_sample, feed_dict=feed_dict)  # NOTE: returns **list**

    def update(self, states, actions, rewards, next_states, flags, sess, logging=True):

        # calculate targets
        feed_dict = {
            self.rewards_pl: rewards,
            self.states_pl: next_states,
            self.flags_pl: ~np.array(flags)
        }
        targets = sess.run(self.targets, feed_dict=feed_dict)

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

class ActorCritic():

    def __init__(
            self,
            lr=0.01,
            entropy_beta=0.01,
            value_beta=0.5,
            gamma=0.99,
            tmax=5,
            max_grad_norm=0.5,
            device='/gpu:0'):

        self.gamma = gamma
        self.tmax = tmax

        # construct graph
        with tf.device(device):

            # placeholders
            states_pl = tf.placeholder(tf.float32, [None, 4], name='states')
            actions_pl = tf.placeholder(tf.int32, [None], name='actions')
            targets_pl = tf.placeholder(tf.float32, [None], name='targets')

            # networks
            values = mlp(states_pl, [64, 32, 1], scope='value')
            policy_logits = mlp(states_pl, [64, 32, 2], scope='policy')
            action_mask = tf.one_hot(actions_pl, int(policy_logits.shape[1]))
            policy = tf.reduce_sum(action_mask * tf.nn.log_softmax(policy_logits), axis=1)

            # actions
            action_sample = tf.squeeze(
                tf.multinomial(logits=policy_logits, num_samples=1),
                axis=1
            )

            # losses
            advantage = targets_pl - tf.stop_gradient(values)
            policy_loss = -tf.reduce_mean(policy * advantage)
            value_loss = tf.reduce_mean(tf.square(targets_pl - values))
            entropy_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.multiply(
                        tf.nn.softmax(policy_logits),  # probabilities
                        tf.nn.log_softmax(policy_logits)  # log probabilities
                    ),
                    axis=1
                )
            )
            loss = policy_loss + entropy_beta * entropy_loss + value_beta * value_loss

            # updates
            optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, decay=0.99, epsilon=1e-5)
            gradients = optimizer.compute_gradients(loss)
            train_op = optimizer.apply_gradients(gradients)

        # tensorboard
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('entropy_loss', entropy_loss)
        tf.summary.scalar('value_loss', value_loss)
        tf.summary.histogram('policy_logits', policy_logits)
        tf.summary.histogram('values', values)
        tf.summary.histogram('advantage', advantage)
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

def bootstrapped_values(terminal_value, rewards, gamma):
    """Calculate targets used to update policy and value functions."""
    targets = []
    R = terminal_value
    for r in rewards[-1::-1]:
        R = r + gamma * R
        targets += [R]
    return targets[-1::-1]  # reverse to match original ordering

class AtariBatchGenerator():

    def __init__(self, envs, agent, sess):
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
            # print(f"step={step}")
            actions = self.agent.select_actions(states, sess)  # *simultaneous* action selection!
            if step == 0:
                init_actions = actions.copy()
            # perform actions env-by-env...
            for idx in range(self.size):
                if not done_flags[idx]:  # check if environment reached end of episode this sample
                    # env = self.envs[idx]
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

                    # logging
                    # print(f"idx={idx}, steps={self.episode_steps[idx]}, return={self.episode_rewards[idx]}, done={done}")

                    # check if we reached end of episode
                    if done:
                        # print(f"environment {idx} reached end of episode!")
                        # print(f"return={episode_rewards[idx]}")
                        # print(f"steps={episode_steps[idx]}")
                        done_flags[idx] = True
                        episode_rewards += [self.episode_rewards[idx]]
                        episode_steps += [self.episode_steps[idx]]
                        # reset environment
                        # print(f"resetting environment...")
                        self.states[idx] = self.envs[idx].reset()
                        self.episode_rewards[idx] = 0.0
                        self.episode_steps[idx] = 0
                else:
                    # print("skipping!")
                    pass

        batch_data = (init_states, init_actions, discounted_rewards, states, done_flags)
        batch_info = batch_steps, len(episode_rewards), episode_rewards, episode_steps
        return batch_data, batch_info

class BatchGenerator():

    def __init__(self, envs, agent, sess):
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
