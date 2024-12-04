import gymnasium as gym
import matplotlib.pyplot as plt

from agent import Agent

################################## TRAINING ##################################

print('\033[32m[Training]\033[0m')

env = gym.make('Pendulum-v1')
agent = Agent(env)
agent.learn(n_episodes=5_000, gamma=.5)

################################# EVALUATION #################################

print('\033[34m[Evaluation]\033[0m')

env = gym.make('Pendulum-v1', render_mode='human')
agent = Agent(env, 'policy_model.pt')

# Plot settings
fig, ax = plt.subplots()
reward_plot, = ax.plot([], [], color='b')
ax.set_xlabel('Timestep')
ax.set_ylabel('Reward')
ax.set_title('Reward evolution during evaluation episode')
ep_rewards = []
ep_timesteps = 0

# Environment initialization
obs, _ = env.reset()
done = trunc = False

# Evaluation episode
while not (done or trunc):
    action = agent.predict(obs)
    obs, reward, done, trunc, _ = env.step(action)

    # Plot reward evolution
    ep_timesteps += 1
    ep_rewards.append(reward)
    reward_plot.set_data(range(1, ep_timesteps + 1), ep_rewards)
    ax.set_xlim(0, ep_timesteps + 1)
    ax.set_ylim(min(ep_rewards) - 1, max(ep_rewards) + 1)
    plt.draw()
    plt.pause(0.01)

plt.ioff()
plt.show()

env.close()
