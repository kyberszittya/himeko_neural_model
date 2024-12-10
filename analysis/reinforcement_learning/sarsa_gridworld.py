import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from PIL import Image

# Gridworld setup
class Gridworld:
    def __init__(self, grid_size, start, goal, obstacles, time_limit):
        self.grid_size = grid_size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.time_limit = time_limit
        self.reset()

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action):
        x, y = self.state
        actions = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # Left, Right, Up, Down
        dx, dy = actions[action]
        new_state = (x + dx, y + dy)

        # Boundary and obstacle check
        if 0 <= new_state[0] < self.grid_size and 0 <= new_state[1] < self.grid_size:
            if new_state not in self.obstacles:
                self.state = new_state

        self.steps += 1
        reward = 1 if self.state == self.goal else -0.1
        done = self.state == self.goal or self.steps >= self.time_limit
        return self.state, reward, done


# SARSA implementation
def sarsa(env, num_episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.grid_size, env.grid_size, 4))  # State-action values
    frames = []
    rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        action = np.random.choice(4) if np.random.rand() < epsilon else np.argmax(Q[state])

        total_reward = 0
        done = False
        while not done:
            next_state, reward, done = env.step(action)
            next_action = np.random.choice(4) if np.random.rand() < epsilon else np.argmax(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            total_reward += reward
            state, action = next_state, next_action

            # Add frame for visualization
            frames.append(plot_grid(env, total_reward, episode))
        rewards.append(total_reward)
    return Q, frames, rewards

# Visualization method
def plot_grid(env, reward, episode):
    grid = np.zeros((env.grid_size, env.grid_size))
    grid[env.goal] = 2  # Goal
    for obs in env.obstacles:
        grid[obs] = -1  # Obstacles
    grid[env.state] = 1  # Agent

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="coolwarm", vmin=-1, vmax=2)
    ax.set_title(f"Episode: {episode + 1} | Reward: {reward:.2f}")
    ax.axis("off")

    # Convert to image and close the plot
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image

# GIF creation function
def create_gif(frames, filename, fps=10):
    image_frames = [Image.fromarray(frame) for frame in frames]
    image_frames[0].save(
        filename,
        save_all=True,
        append_images=image_frames[1:],
        duration=1000 // fps,
        loop=0,
    )

# Moving average smoothing function
def moving_average(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    return smoothed_data

# Visualization of smoothed rewards
def plot_smoothed_rewards(rewards, smoothed_rewards, window_size):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label="Original Rewards", alpha=0.5)
    plt.plot(range(window_size - 1, len(rewards)), smoothed_rewards, label=f"Smoothed Rewards (Window={window_size})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Smoothing Using Moving Average")
    plt.legend()
    plt.grid()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Parameters
    grid_size = 5
    start = (0, 0)
    goal = (4, 4)
    obstacles = [(1, 1), (2, 2), (3, 3)]
    time_limit = 20
    num_episodes = 200
    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1

    # Create environment
    env = Gridworld(grid_size, start, goal, obstacles, time_limit)

    # Train SARSA and collect frames
    _, frames, rewards = sarsa(env, num_episodes, alpha, gamma, epsilon)

    # Save to GIF
    #create_gif(frames, "gridworld_sarsa.gif")
    smoothing_window = 10
    smoothed_rewards = moving_average(rewards, smoothing_window)
    plot_smoothed_rewards(rewards, smoothed_rewards, smoothing_window)