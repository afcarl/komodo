import gym
import numpy as np
import tensorflow as tf
from versterken.keras import clip_by_norm

class ActorCritic():

    def __init__(
            self,
            placeholders,
            networks,
            lr=0.001,
            entropy_beta=0.01,
            update_freq=4,
            gamma=0.99,
            history=4,
            clip_norm=0.1,
            device='/gpu:0',
            atari=True
        ):

        self.update_freq = update_freq
        self.gamma = gamma
        self.history = history
        self.atari = atari

        # construct graph
        with tf.device(device):
            # placeholders
            states_pl = placeholders['states']
            actions_pl = placeholders['actions']
            rewards_pl = placeholders['rewards']
            flags_pl = placeholders['flags']
            targets_pl = placeholders['targets']

            # networks
            values = networks['value']
            policy_logits = networks['policy']
            action_mask = tf.one_hot(actions_pl, int(policy_logits.shape[1]))
            policy = tf.reduce_sum(action_mask * tf.nn.log_softmax(policy_logits), axis=1)

            # targets
            targets = rewards_pl + flags_pl * self.gamma ** self.history * values

            # actions
            action_sample = tf.squeeze(
                tf.multinomial(logits=policy_logits, num_samples=1),
                axis=1
            )

            # losses
            policy_loss = -tf.reduce_mean(policy * (targets_pl - tf.stop_gradient(values)))
            value_loss = 0.5 * tf.reduce_mean(tf.square(targets_pl - values))
            entropy_loss = entropy_beta * tf.reduce_mean(
                tf.multiply(
                    tf.nn.softmax(policy_logits),  # probabilities
                    tf.nn.log_softmax(policy_logits)  # log probabilities
                )
            )
            loss = policy_loss + entropy_loss + value_loss

            # updates
            optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-3)
            gradients = clip_by_norm(optimizer.compute_gradients(loss), clip_norm=clip_norm)
            train_op = optimizer.apply_gradients(gradients)

        # tensorboard
        tf.summary.scalar('gradient_norm', tf.norm(gradients[0][0]))
        tf.summary.scalar('entropy', -entropy_loss / entropy_beta)
        tf.summary.histogram('policy', policy_logits)
        tf.summary.histogram('value', values)
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

    def action(self, state, sess):
        if self.atari:
            feed_dict = {self.states_pl: states.reshape(-1, 84, 84, self.history)}
        else:
            feed_dict = {self.states_pl: states.reshape(1, -1)}
        return sess.run(self.action_sample, feed_dict=feed_dict)[0]  # NOTE: return **scalar**!

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

def bootstrapped_values(terminal_value, rewards, gamma):
    """Calculate targets used to update policy and value functions."""
    targets = []
    R = terminal_value
    for r in rewards[-1::-1]:
        R = r + gamma * R
        targets += [R]
    return targets[-1::-1]  # reverse to match original ordering

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
        init_states = self.states
        states = self.states  # will keep track of terminal states...
        discounted_rewards = [0.0] * self.size

        # batch episode data
        episode_rewards = []
        episode_steps = []

        # other batch info
        batch_steps = 0

        # generate trajectories
        done_flags = [False] * self.size  # True if environment reaches end of epsidoe this sample
        for step in range(n):
            # print(f"step={step}")
            actions = self.agent.select_actions(states, sess)  # *simultaneous* action selection!
            if step == 0:
                init_actions = actions.copy()
            # perform actions env-by-env...
            for idx in range(self.size):
                if not done_flags[idx]:  # check if environment reached end of episode this sample
                    env = self.envs[idx]
                    action = actions[idx]
                    state, reward, done, info = env.step(action)

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
                        self.states[idx] = env.reset()
                        self.episode_rewards[idx] = 0.0
                        self.episode_steps[idx] = 0
                else:
                    # print("skipping!")
                    pass

        batch_data = (init_states, init_actions, discounted_rewards, states, done_flags)
        batch_info = batch_steps, len(episode_rewards), episode_rewards, episode_steps
        return batch_data, batch_info
