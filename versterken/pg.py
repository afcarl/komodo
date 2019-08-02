import tensorflow as tf

class PolicyGradient():

    def __init__(self):
        pass

    def action(self, state, sess):
        return sess.run(self.action_sample, feed_dict={self.states_pl: state})[0]

    def targets(self, rewards):
        return rewards_to_go(rewards, discount_rate)

    def update(self, batches, sess):
        """Perform updates in `batch_size` batches.

            - `batches = [batch, batch, ..., batch]`

            Each batch has fixed length, except possibly the last batch.

            - `batch = [states, actions, targets]`
        """

        summaries = []
        for batch in batches:
            states, actions, targets = split_batch(batch)
            summary, _, _ = sess.run(
                [
                    self.summary_op,
                    self.policy_update,
                ],
                feed_dict={
                    self.states_pl: states,
                    self.actions_pl: actions,
                    self.targets_pl: targets
                }
            )
            summaries += [summary]
        return summaries

    def save(self, path, step, sess):
        self.saver.save(sess, save_path=path, global_step=step)
