import tensorflow as tf
from versterken.atari import rgb_to_grayscale

class ActorCritic():

    def __init__(
            self,
            placeholders,
            networks,
            lr=0.001,
            beta=0.,
            update_freq=5,
            gamma=1.,
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
            value_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(value_loss)
            policy_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(policy_loss + entropy_loss)

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

        # preprocessor
        self.frame_pl = tf.placeholder(tf.uint8, [210, 160, 3])
        self.preprocess_op = rgb_to_grayscale(self.frame_pl)  # uint8, 84 x 84

    def action(self, state, sess):
        return sess.run(self.action_sample, feed_dict={self.states_pl: state})[0]

    def targets(self, terminal_state, rewards, done, sess):
        if done:
            terminal_value = 0.0
        else:
            terminal_value = sess.run(
                self.values,
                feed_dict={
                    self.states_pl: terminal_state
                }
            )
        return bootstrapped_values(terminal_value, rewards, self.gamma)

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
        self.states = []
        self.actions = []
        self.rewards = []
        self.state = self.env.reset()

    def sample(self, n):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        for i in range(n):
            state = self.state
            action = self.agent.action(self.state)  # TODO: reshape state
            next_state, reward, done, info = self.env.step(action)
            self.states += [state]
            self.actions += [action]
            self.rewards += [reward]
            self.state = next_state
            if done:
                self.state = self.env.reset()
                break
        return self.states, self.actions, self.rewards, next_state, done
