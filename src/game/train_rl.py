import torch
import torch.nn as nn
import torch.optim as optim
import math, random
import matplotlib.pyplot as plt
import numpy as np

from .levels import generate_random_level, load_level
from .ai import _simulate_shot
from .entities import create_world, create_ground


# RL AI is a policy gradient agent.

class RLPolicy(nn.Module):
    def __init__(self):
        super(RLPolicy, self).__init__()
        self.fc1 = nn.Linear(42, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        mean_a, log_std_a, mean_f, log_std_f = x
        return mean_a, log_std_a, mean_f, log_std_f

# environment simulation takes a given level and simulates the shot and returns the reward i.e. did the bird hit the target
def run_env_simulation(space, target, angle_deg, impulse_magnitude):
    """Simulate the shot and return the reward"""
    hit, distance_to_target = _simulate_shot(space, target, angle_deg, impulse_magnitude)
    
    return (1.0 if hit else 0.0, distance_to_target)
   

def initialize_env():
    space = create_world((0, 900))
    ground = create_ground(space, y=500, width=960)
    episode_level = generate_random_level()
    bird, targets, obstacles = load_level(space, episode_level, use_llm=False)
    return space, bird, targets, obstacles


def flatten_normalize_state(state):
    ### flatten and normalize the state to the form [target_x, target_y, obs1_x, obs1_y, obs1_w, obs1_h, obs2_x, obs2_y, obs2_w, obs2_h, ...]
    MAX_TARGETS = 1
    MAX_OBSTACLES = 10
    MAX_X = 960
    MAX_Y = 540
    
    space, bird, targets, obstacles = state
    features = []
    for target in targets:
        features.append(target.body.position.x / MAX_X)
        features.append(target.body.position.y / MAX_Y)
    for obs, size in obstacles:
        features.append(obs.body.position.x / MAX_X)
        features.append(obs.body.position.y / MAX_Y)
        features.append(size[0] / MAX_X)
        features.append(size[1] / MAX_Y)

    ### pad the features to the MAX_TARGETS and MAX_OBSTACLES
    features = features[:MAX_TARGETS * 2 + MAX_OBSTACLES * 4]
    features = features + [0] * (MAX_TARGETS * 2 + MAX_OBSTACLES * 4 - len(features))
    
    ### convert the features to a tensor
    feature_vec = torch.tensor(features, dtype=torch.float32)
    return feature_vec

def get_action(policy, state):
    
    feature_vec = flatten_normalize_state(state)
    
    max_angle = 85
    max_impulse_magnitude = 2000
    # Get policy distribution
    mean_a, log_std_a, mean_f, log_std_f = policy(feature_vec)
    
    # Clamp log_std to prevent numerical instability
    # This ensures std is between exp(-5) ≈ 0.007 and exp(2) ≈ 7.4
    log_std_a = torch.clamp(log_std_a, -5, 2)
    log_std_f = torch.clamp(log_std_f, -5, 2)
    
    std_a = log_std_a.exp()
    std_f = log_std_f.exp()
    
    # Check for NaN values and replace with safe defaults
    if torch.isnan(mean_a) or torch.isnan(std_a):
        mean_a = torch.tensor(0.0)
        std_a = torch.tensor(1.0)
    if torch.isnan(mean_f) or torch.isnan(std_f):
        mean_f = torch.tensor(0.0)
        std_f = torch.tensor(1.0)

    dist_a = torch.distributions.Normal(mean_a, std_a)
    dist_f = torch.distributions.Normal(mean_f, std_f)

    # Sample action
    angle_deg = dist_a.sample()
    impulse_magnitude = dist_f.sample()

    angle_deg_scaled = torch.tanh(angle_deg) * (max_angle / 2) + (max_angle / 2)
    impulse_magnitude_scaled = torch.tanh(impulse_magnitude) * (max_impulse_magnitude / 2) + (max_impulse_magnitude / 2)
    
    return angle_deg, angle_deg_scaled, impulse_magnitude, impulse_magnitude_scaled, dist_a, dist_f, feature_vec


def update_training_plot(fig, axes, episodes, rewards, losses, angles_scaled, impulses_scaled, 
                        mean_angles, std_angles, mean_impulses, std_impulses, grad_norms_history):
    """Update the existing plot with new data"""
    
    # Clear all axes
    for ax in axes.flat:
        ax.clear()
    
    # Plot 1: Rewards over time (scatter plot)
    axes[0, 0].scatter(episodes, rewards, alpha=0.6, s=10, c='b')
    axes[0, 0].set_title('Rewards per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Loss over time
    axes[0, 1].plot(episodes, losses, 'r-', alpha=0.7, linewidth=1)
    axes[0, 1].set_title('Loss per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Angle distribution over time
    axes[0, 2].scatter(episodes, angles_scaled, alpha=0.6, s=10, c='g')
    axes[0, 2].set_title('Angle Distribution')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Angle (degrees)')
    axes[0, 2].set_ylim(0, 90)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Impulse distribution over time
    axes[1, 0].scatter(episodes, impulses_scaled, alpha=0.6, s=10, c='orange')
    axes[1, 0].set_title('Impulse Distribution')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Impulse Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Policy mean and std for angles
    axes[1, 1].plot(episodes, mean_angles, 'b-', label='Mean', linewidth=2)
    axes[1, 1].fill_between(episodes, 
                           np.array(mean_angles) - np.array(std_angles),
                           np.array(mean_angles) + np.array(std_angles),
                           alpha=0.3, label='±1 std')
    axes[1, 1].set_title('Angle Policy (Mean ± Std)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Angle (raw)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Policy mean and std for impulses
    axes[1, 2].plot(episodes, mean_impulses, 'r-', label='Mean', linewidth=2)
    axes[1, 2].fill_between(episodes, 
                           np.array(mean_impulses) - np.array(std_impulses),
                           np.array(mean_impulses) + np.array(std_impulses),
                           alpha=0.3, label='±1 std')
    axes[1, 2].set_title('Impulse Policy (Mean ± Std)')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Impulse (raw)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Plot 7: Gradient norms over time
    axes[2, 0].plot(episodes, grad_norms_history, 'purple', alpha=0.7, linewidth=1)
    axes[2, 0].set_title('Total Gradient Norm')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Gradient Norm')
    axes[2, 0].set_yscale('log')  # Log scale for better visualization
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 8: Angle standard deviation over time
    axes[2, 1].plot(episodes, std_angles, 'g-', alpha=0.7, linewidth=1)
    axes[2, 1].set_title('Angle Policy Std Dev')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Std Dev')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 9: Impulse standard deviation over time
    axes[2, 2].plot(episodes, std_impulses, 'orange', alpha=0.7, linewidth=1)
    axes[2, 2].set_title('Impulse Policy Std Dev')
    axes[2, 2].set_xlabel('Episode')
    axes[2, 2].set_ylabel('Std Dev')
    axes[2, 2].grid(True, alpha=0.3)
    
    # Update the main title
    fig.suptitle(f'RL Training Metrics - Episode {episodes[-1]}', fontsize=16)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # Very brief pause to allow plot to update


#  train the RL AI for a given number of steps, and return the trained model
def train_rl_ai(num_episodes: int, plot_interval: int = 100, curriculum_switch: int = 5000):
    """Train the RL AI for a given number of steps"""  
    policy = RLPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    
    # Initialize tracking lists
    episodes = []
    rewards = []
    losses = []
    angles_scaled = [] 
    impulses_scaled = []
    mean_angles = []
    std_angles = []
    mean_impulses = []
    std_impulses = []
    grad_norms_history = []
    
    # Initialize the plot with 3x3 subplots
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.suptitle('RL Training Metrics - Starting...', fontsize=16)
    plt.tight_layout()
    plt.show(block=False)
    
    sum_reward = 0
    for episode_idx in range(num_episodes):
        state = initialize_env()
        angle_deg, angle_deg_scaled, impulse_magnitude, impulse_magnitude_scaled, dist_a, dist_f, feature_vec = get_action(policy, state)

        space, bird, targets, obstacles = state
        target = targets[0]

        reward, distance_to_target = run_env_simulation(space, target, float(angle_deg_scaled.item()), float(impulse_magnitude_scaled.item()))
        
        if episode_idx < curriculum_switch:
            if reward > 0:
                reward = reward
            else:
                reward = 1-distance_to_target   
        else:
            reward = reward
        reward = torch.tensor(reward, dtype=torch.float32)

        # Compute policy loss (REINFORCE)
        log_prob = dist_a.log_prob(angle_deg) + dist_f.log_prob(impulse_magnitude)
        loss = -log_prob * reward
        sum_reward += reward

        # Track metrics
        episodes.append(episode_idx)
        rewards.append(reward.item())
        losses.append(loss.item())
        angles_scaled.append(angle_deg_scaled.item())
        impulses_scaled.append(impulse_magnitude_scaled.item())
        mean_angles.append(dist_a.mean.item())
        std_angles.append(dist_a.scale.item())
        mean_impulses.append(dist_f.mean.item())
        std_impulses.append(dist_f.scale.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate gradient norms for tracking
        grad_norms = []
        for param in policy.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
            else:
                grad_norms.append(0.0)
        grad_norms_history.append(sum(grad_norms))

        if episode_idx % 100 == 0 and reward > 0:
            print(f'--------------------------------')
            print(f"Episode {episode_idx}, reward={reward}, loss={loss}, distance_to_target={distance_to_target}, sum_reward={sum_reward}")
            print(f'angle_deg: {angle_deg}, angle_deg_scaled: {angle_deg_scaled}, impulse_magnitude: {impulse_magnitude}, impulse_magnitude_scaled: {impulse_magnitude_scaled}')
            print(f'mean_a: {dist_a.mean}, std_a: {dist_a.scale}, mean_f: {dist_f.mean}, std_f: {dist_f.scale}')
            print(f'feature_vec: {feature_vec}')
            print(f'target: {[target.body.position.x, target.body.position.y]}')
            print(f'obstacles: {[[obs.body.position.x, obs.body.position.y] for obs, size in obstacles]}')
            print(f'gradient norms: {grad_norms}')
            print(f'total gradient norm: {sum(grad_norms):.6f}')
            
            sum_reward = 0
            
        # Update plot at specified intervals
        if episode_idx % plot_interval == 0:
            update_training_plot(fig, axes, episodes, rewards, losses, angles_scaled, impulses_scaled,
                               mean_angles, std_angles, mean_impulses, std_impulses, grad_norms_history)
    
    # Keep the plot open at the end
    plt.ioff()  # Turn off interactive mode
    plt.show()
    
    # Print final summary
    print(f"\n=== Final Training Summary ===")
    print(f"Total Episodes: {len(episodes)}")
    print(f"Final Reward: {rewards[-1]:.3f}")
    print(f"Average Reward (last 100): {np.mean(rewards[-100:]):.3f}")
    print(f"Final Angle Mean: {mean_angles[-1]:.3f} ± {std_angles[-1]:.3f}")
    print(f"Final Impulse Mean: {mean_impulses[-1]:.3f} ± {std_impulses[-1]:.3f}")
    print(f"Angle Range: {min(angles_scaled):.1f} - {max(angles_scaled):.1f} degrees")
    print(f"Impulse Range: {min(impulses_scaled):.1f} - {max(impulses_scaled):.1f}")
    
    return policy


if __name__ == "__main__":
    train_rl_ai(10000, plot_interval=100, curriculum_switch=5000)