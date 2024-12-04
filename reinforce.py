import torch
import torch.nn as nn
import torch.distributions as dist

import matplotlib.pyplot as plt

from tqdm import tqdm


class Policy(nn.Module):
    '''
    Policy network.
    '''

    def __init__(self, input_size, output_size, learning_rate=1e-4):
        '''
        Policy network definition.
        '''
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)

        # Mean of the action distribution
        self.fc3_mean = nn.Linear(64, output_size)

        # Standard deviation
        self.fc3_std = nn.Linear(64, output_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, epsilon=1e-4):
        '''
        Forward pass.
        Note: epsilon avoids NaN std
        '''
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        mean = self.fc3_mean(x)
        std = torch.log(1 + torch.exp(self.fc3_std(x))) + epsilon

        return mean, std


class ExperienceBuffer:
    '''
    Experience buffer class.
    '''

    def __init__(self, max_size=10_000):
        '''
        Creates an empty buffer.
        '''
        self.buffer = []
        self.size = 0
        self.max_size = max_size

    def store(self, state, action, reward):
        '''
        Stores a tuple (s, a, r).
        '''
        self.buffer.append((state, action, reward))
        self.size += 1
        if self.size > self.max_size:
            self.buffer.pop(0)

    def get_all(self):
        '''
        Returns the full buffer.
        '''
        return self.buffer

    def clear(self):
        '''
        Empties the buffer.
        '''
        self.buffer.clear()
        self.size = 0

    def __iter__(self):
        '''
        Buffer iterator.
        '''
        return iter(self.buffer)

    def __str__(self):
        '''
        Buffer str representation.
        '''
        buffer_str = ''
        for elem in self.buffer:
            buffer_str += f'{elem[0]}\t{elem[1]}\t{elem[2]}\n'
        return buffer_str


class Agent:
    '''
    Agent class for the REINFORCE algorithm.
    '''

    def __init__(self, env, policy_model=None):
        '''
        Agent initialization.
        '''

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.env = env
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.buffer = ExperienceBuffer()

        self.policy = Policy(
            self.obs_space.shape[0], self.action_space.shape[0]).to(self.device)

        if policy_model:
            self.policy.load_state_dict(torch.load(
                policy_model, map_location=self.device, weights_only=True))
            self.policy.eval()
            print(f'Loaded model {policy_model}')

        self.alpha = None  # learning rate
        self.gamma = None  # discount factor

    def predict(self, obs):
        '''
        Returns an action sampled from a distribution based on the current observation.
        '''

        # Get mean and std from policy
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
        mean, std = self.policy(obs_tensor)

        # Create distribution and sample action
        distrib = dist.Normal(mean, std)
        action = distrib.sample()

        # Clip action to the action space
        min_action = torch.tensor(
            self.action_space.low, dtype=torch.float32).to(self.device)
        max_action = torch.tensor(
            self.action_space.high, dtype=torch.float32).to(self.device)
        action = torch.clamp(action, min_action, max_action)

        return action.detach().cpu().numpy()

    def learn(self, n_episodes, alpha=1e-4, gamma=.9, model_name="policy_model"):
        '''
        REINFORCE algorithm implementation for agent learning.
        '''

        self.alpha = alpha
        self.gamma = gamma

        # 1. Initialize policy
        self.policy = Policy(
            self.obs_space.shape[0], self.action_space.shape[0], learning_rate=alpha).to(self.device)

        # (Reward plot settings)
        plt.ion()
        _, ax = plt.subplots()
        reward_plot, = ax.plot([], [], color='g')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episodic cumulative reward during training')
        ep_cum_rewards = []

        # 2. For each episode...
        for _ in tqdm(range(n_episodes), desc="Training Episodes", ncols=100, unit="episode"):

            total_ep_reward = 0

            # 2.1. Initialize environment
            state, _ = self.env.reset()
            done = trunc = False

            # 2.2. Run full episode
            while not (done or trunc):
                # 2.2.1. Get action from policy
                action = self.predict(state)
                # 2.2.2. Execute action. Get new state and reward
                next_state, reward, done, trunc, _ = self.env.step(action)
                # 2.2.3. Save tuples (s, a, r)
                self.buffer.store(state, action, reward)

                state = next_state

                total_ep_reward += reward

            ep_cum_rewards.append(total_ep_reward)

            # 2.3. Compute discounted return for each step
            G = 0
            returns = []
            for _, _, reward in reversed(self.buffer.get_all()):
                G = reward + self.gamma * G
                returns.insert(0, G)

            # 2.4. Policy update
            for (obs, action, _), G in zip(self.buffer.get_all(), returns):
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32).to(self.device)
                action_tensor = torch.tensor(
                    action, dtype=torch.float32).to(self.device)

                # 2.4.1. Get action mean and std, then create normal distribution
                mean, std = self.policy(obs_tensor)
                distrib = dist.Normal(mean, std)

                # 2.4.2. Get action log probability
                log_prob = distrib.log_prob(action_tensor).sum()

                # 2.4.3. Compute REINFORCE loss
                loss = -log_prob * G

                # 2.4.2. Backpropagation
                self.policy.optimizer.zero_grad()
                loss.backward()
                self.policy.optimizer.step()

            # 2.5. Clear buffer
            self.buffer.clear()

            # (Update plot)
            reward_plot.set_data(
                range(1, len(ep_cum_rewards) + 1), ep_cum_rewards)
            ax.set_xlim(0, len(ep_cum_rewards) + 1)
            ax.set_ylim(min(ep_cum_rewards) - 1, max(ep_cum_rewards) + 1)
            plt.draw()
            plt.pause(0.01)

        # 3. Save learned model
        torch.save(self.policy.state_dict(), model_name + '.pt')
        print('Model saved to {model_name}.pt')

        plt.ioff()
