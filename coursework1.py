def get_CID():
  return "06005311" # Return your CID (add 0 at the beginning to ensure it is 8 digits long)

def get_login():
  return "eg424" # Return your short imperial login

# DP Agent
class DP_agent(object):
    def solve(self, env):
 
        # Initialisation
        state_size, action_size = env.get_state_size(), env.get_action_size()
        policy = np.zeros((state_size, action_size))
        V = np.zeros(state_size)
        T = env.get_T()
        R = env.get_R()
        gamma = env.get_gamma()
        theta = 1e-6

        # Policy Iteration loop
        while True:
            # Policy Evaluation
            while True:
                delta = 0
                for state in range(state_size):
                    v = V[state]
                    action = np.argmax(policy[state])
                    # Calculate value of state under current policy
                    V[state] = np.sum(T[state, :, action] * (R[state, :, action] + gamma * V))
                    delta = max(delta, abs(v - V[state]))  # Track max change across all states

                if delta < theta:
                    break

            # Policy Improvement
            policy_stable = True
            for state in range(state_size):
                # Calculate action values for all actions
                old_action = np.argmax(policy[state])
                action_values = np.zeros(action_size)
                for action in range(action_size):
                    action_values[action] = np.sum(T[state, :, action] * (R[state, :, action] + gamma * V))

                # Find best action based on current value function
                best_action = np.argmax(action_values)

                # Update policy
                if old_action != best_action:
                    policy_stable = False
                policy[state] = 0
                policy[state, best_action] = 1

            if policy_stable:
                return policy, V

# MC Agent
class MC_agent(object):
    def __init__(self, epsilon=0.7, decay_rate=0.999, alpha=0.5, gamma=0.82, num_episodes=400):
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.Q = None
        self.policy = None
        self.visit_count = None


    def solve(self, env):
        # Initialisation
        state_size, action_size = env.get_state_size(), env.get_action_size()
        Q = np.ones((state_size, action_size))
        visit_count = np.zeros((state_size, action_size))
        policy = np.zeros((state_size, action_size))

        values, total_rewards = [], []

        for episode in range(self.num_episodes):
            # Generate an episode
            _, state, _, _ = env.reset()
            episode_data = []
            done = False
            while not done:
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(action_size)  # Explore
                else:
                  action = np.argmax(Q[state])  # Exploit
                _, next_state, reward, done = env.step(action)
                episode_data.append((state, action, reward))
                state = next_state

            # Returns calculation
            states, actions, rewards = zip(*episode_data)
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            # Q-value update
            visited_pairs = set()
            for t, (state, action) in enumerate(zip(states, actions)):
                if (state, action) not in visited_pairs:
                    visited_pairs.add((state, action))
                    visit_count[state, action] += 1
                    decaying_alpha = 1 / visit_count[state, action]
                    Q[state, action] += decaying_alpha * (returns[t] - Q[state, action])

            # Policy update
            policy = np.zeros((state_size, action_size))
            for state in range(state_size):
                best_action = np.argmax(Q[state])
                policy[state] = self.epsilon / action_size
                policy[state, best_action] += (1 - self.epsilon)

            # Epsilon decay
            self.epsilon *= self.decay_rate

            values.append(np.max(Q, axis=1))
            total_rewards.append(sum(rewards))

        return policy, values, total_rewards

