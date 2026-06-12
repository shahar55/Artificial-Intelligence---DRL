import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

# ==========================================
# GLOBAL HYPERPARAMETERS (Constants)
# ==========================================

LR = 1e-4                  # Learning rate for the Adam optimizer
GAMMA = 0.99               # Discount factor for future rewards
UPDATE_TARGET = 500        # Frequency (in optimization steps) to sync target network
BUFFER_CAPACITY = 50000   # Total capacity of the prioritized replay buffer
ALPHA = 0.6                # PER prioritization exponent factor
BETA_START = 0.4           # Initial value of the PER importance-sampling exponent
TOTAL_EPISODES = 1000      # Total number of training episodes
BATCH_SIZE = 128            # Size of mini-batches sampled from the buffer
EPSILON_START = 1.0        # Initial exploration probability (starts at 100% random actions)
EPSILON_MIN = 0.01         # Minimum exploration probability limit (stops decaying at 1%)
EPSILON_DECAY = 0.995      # Decay rate per episode for epsilon-greedy policy

# ==========================================
# 1. Dueling Q-Network
# ==========================================
class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()

        # Shared trunk representation layers
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Dueling combine formula: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Dimension 1 handles batch broadcasting in PyTorch correctly
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

# ==========================================
# 2. Prioritized Experience Replay
# ==========================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # 1. Calculate sampling probabilities based on priorities: P(i) = p_i^alpha / sum(p_i^alpha)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 2. Calculate Importance-Sampling weights and normalize them: w_i = (N * P(i))^-beta
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalization stabilizes the gradient descent updates

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, np.array(weights))

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

# ==========================================
# 3. D3QN Agent
# ==========================================
class D3QNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.update_counter = 0            # Tracking optimization steps performed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.memory = PrioritizedReplayBuffer(BUFFER_CAPACITY, ALPHA)

    # =========================================================================
    # 🏎️ EPSILON-GREEDY ACTION SELECTION (EXPLORATION VS EXPLOITATION)
    #
    # To find optimal strategies, an RL agent must balance:
    # 1. Exploration (Trying random actions to discover new high-reward states).
    # 2. Exploitation (Leveraging learned knowledge to maximize reward).
    #
    # We roll a random number between 0 and 1:
    # - If random > epsilon: EXPLOIT (Query the neural network for the best action).
    # - If random <= epsilon: EXPLORE (Pick a completely random environment action).
    # =========================================================================
    def act(self, state, epsilon):
        if random.random() > epsilon:
            # 🧠 EXPLOITATION MODE
            # Convert state array to a PyTorch tensor and run it through the network
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(state)
            # Choose the action with the highest predicted Q-value (Greedy action)
            return q_values.argmax().item()
        else:
            # 🎲 EXPLORATION MODE
            # Draw a completely uniform random integer within the action space range
            return random.randrange(self.action_dim)
    # =========================================================================

    def update(self, beta):
        if len(self.memory.buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE, beta)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        # Get current Q predictions for chosen actions
        q_values = self.online_net(states).gather(1, actions)

        # =========================================================================
        # 🎯 DOUBLE DQN TARGET CALCULATION
        #
        # In standard DQN, we select and evaluate the max next action using the
        # same network (target_net), leading to Overestimation Bias.
        # Double DQN decouples this process into two steps:
        # 1. Action Selection: Done by the ONLINE network.
        # 2. Action Evaluation: Done by the TARGET network.
        # =========================================================================
        with torch.no_grad():
            # Step 1: Use the Online Network to select the best action index for the next state
            best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)

            # Step 2: Use the Target Network to evaluate the Q-value of that selected action
            next_q_values = self.target_net(next_states).gather(1, best_next_actions)

            # Step 3: Compute the classic Bellman equation TD target value (masking out terminal states)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)
        # =========================================================================

        # Calculate TD errors for prioritizing updates in the replay buffer
        td_errors = target_q_values - q_values
        new_priorities = (torch.abs(td_errors) + 1e-5).cpu().detach().numpy().flatten()
        self.memory.update_priorities(indices, new_priorities)

        # Calculate squared loss scaled by the importance sampling weights
        loss = (weights * (td_errors ** 2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
       #torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Advance optimization step counter
        self.update_counter += 1

        TAU = 0.005

        # Soft update target network
        for target_param, online_param in zip(self.target_net.parameters(), self.online_net.parameters()):
          target_param.data.copy_(TAU * online_param.data + (1.0 - TAU) * target_param.data)

        return loss.item()

# ==========================================
# 4. Training Loop
# ==========================================
def main():
    env = gym.make("LunarLander-v3")

    state, info = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = D3QNAgent(state_dim, action_dim)

    epsilon = EPSILON_START
    rewards_history = []

    print("Starting D3QN Training (with Hard Target Updates) on LunarLander...")

    for ep in range(TOTAL_EPISODES):
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.act(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)

            agent.memory.push(state, action, reward, next_state, done or truncated)
            state = next_state
            episode_reward += reward

            # Linearly annealing beta exponent parameter to 1.0 throughout training
            beta = min(1.0, BETA_START + ep * (1.0 - BETA_START) / TOTAL_EPISODES)
            agent.update(beta)

        # =========================================================================
        # ⏱️ EPSILON DECAY ANNEALING
        #
        # As training progresses across episodes, the agent becomes more confident
        # in its learned Q-values. We decay epsilon (multiply by EPSILON_DECAY)
        # at the end of every episode to shift weight from exploring to exploiting.
        # We enforce EPSILON_MIN to maintain a 1% baseline floor of eternal curiosity.
        # =========================================================================
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        # =========================================================================

        rewards_history.append(episode_reward)

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {ep+1}/{TOTAL_EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.2f}")

    # Serialize model weights to file
    torch.save(agent.online_net.state_dict(), "d3qn_lunar_model.pth")
    print("\nTraining finished. Model saved to: d3qn_lunar_model.pth")

    # Save training curve diagram
    plt.plot(rewards_history)
    plt.title("D3QN on LunarLander (Hard Updates)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("lunar_training_results.png")

    env.close()


if __name__ == "__main__":
    main()