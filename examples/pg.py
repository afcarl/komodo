

def create_batches(states, actions, rewards, batch_size):
    batches = []
    targets = rewards_to_go(rewards)
    while idx < len(states):
        batch_states = states[idx:idx + batch_size]
        batch_actions = actions[idx:idx + batch_size]
        batch_targets = targets[idx:idx + batch_size]
        batches += [batch_states, batch_actions, batch_targets]
        idx += batch_size
    return batches

def run_episode(env, agent):
    batches = []
    states = []
    actions = []
    rewards = []
    steps = 0
    start = time.time()
    while True:
        # action = ...
        next_state, reward, done, info = env.step(action)
        states += [state]
        actions += [action]
        rewards += [reward]
        state = next_state
        steps += 1
        if done:
            break
    batches = create_batches(
        states,
        actions,
        rewards,
        batch_size
    )
    stop = time.time()
    return batches, sum(rewards), steps, stop - start

def train():

    while episodes < max_episodes:

        # run an episode

        # perform update(s)

        # logging

        # checkpoint

if __name__ == "__main__":
    train()
