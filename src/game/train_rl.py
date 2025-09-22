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
        self.fc1 = nn.Linear(57, 64)
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
   

def initialize_env(MAX_OBSTACLES=10):
    space = create_world((0, 900))
    ground = create_ground(space, y=500, width=960)
    episode_level = generate_random_level(MAX_OBSTACLES)
    bird, targets, obstacles = load_level(space, episode_level, use_llm=False)
    return space, bird, targets, obstacles


def flatten_normalize_state(state):
    MAX_TARGETS = 1
    MAX_OBSTACLES = 10
    MAX_X = 960
    MAX_Y = 540
    
    space, bird, targets, obstacles = state
    assert len(obstacles) <= MAX_OBSTACLES, f"Number of obstacles is greater than MAX_OBSTACLES: {len(obstacles)} > {MAX_OBSTACLES}"

    features = []
    
    # 1. Bird position (2 features)
    features.append(bird.body.position.x / MAX_X)
    features.append(bird.body.position.y / MAX_Y)
    
    # 2. Target position (2 features)
    for target in targets:
        features.append(target.body.position.x / MAX_X)
        features.append(target.body.position.y / MAX_Y)
    
    # 3. Relative target information (4 features)
    if targets:
        target = targets[0]
        dx = target.body.position.x - bird.body.position.x
        dy = target.body.position.y - bird.body.position.y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)
        
        features.append(dx / MAX_X)  # Relative x
        features.append(dy / MAX_Y)  # Relative y
        features.append(distance / math.sqrt(MAX_X**2 + MAX_Y**2))  # Normalized distance
        features.append(angle_to_target / math.pi)  # Normalized angle [-1, 1]
    
    # 4. Obstacle information (4 * MAX_OBSTACLES features)
    for obs, size in obstacles:
        features.append(obs.body.position.x / MAX_X)
        features.append(obs.body.position.y / MAX_Y)
        features.append(size[0] / MAX_X)
        features.append(size[1] / MAX_Y)
    
    # 5. NEW: Obstacle density features (3 features)
    # Count obstacles in different regions relative to bird
    left_obstacles = sum(1 for obs, _ in obstacles if obs.body.position.x < bird.body.position.x)
    right_obstacles = sum(1 for obs, _ in obstacles if obs.body.position.x >= bird.body.position.x)
    total_obstacles = len(obstacles)
    
    features.append(left_obstacles / MAX_OBSTACLES)  # Normalized count
    features.append(right_obstacles / MAX_OBSTACLES)  # Normalized count
    features.append(total_obstacles / MAX_OBSTACLES)  # Normalized count
    
    # 6. NEW: Target-obstacle relationships (2 features)
    if targets and obstacles:
        target = targets[0]
        # Find closest obstacle to target
        min_dist_to_obstacle = float('inf')
        for obs, _ in obstacles:
            dist = math.sqrt((obs.body.position.x - target.body.position.x)**2 + 
                           (obs.body.position.y - target.body.position.y)**2)
            min_dist_to_obstacle = min(min_dist_to_obstacle, dist)
        
        features.append(min_dist_to_obstacle / math.sqrt(MAX_X**2 + MAX_Y**2))  # Normalized distance
        
        # Check if target is "blocked" by obstacles
        target_blocked = 0
        for obs, size in obstacles:
            # Simple check: if obstacle is between bird and target
            obs_x, obs_y = obs.body.position.x, obs.body.position.y
            target_x, target_y = target.body.position.x, target.body.position.y
            bird_x, bird_y = bird.body.position.x, bird.body.position.y
            
            # Check if obstacle is in the line of sight
            if (min(bird_x, target_x) <= obs_x <= max(bird_x, target_x) and 
                min(bird_y, target_y) <= obs_y <= max(bird_y, target_y)):
                target_blocked = 1
                break
        
        features.append(target_blocked)  # Binary feature
    else:
        features.append(1.0)  # No obstacles = "not blocked"
        features.append(0.0)  # No obstacles = not blocked
    
    # 7. NEW: Game state features (2 features)
    # Episode progress (if you want to add curriculum learning)
    features.append(0.0)  # Placeholder for episode progress
    
    # Difficulty indicator (based on number of obstacles)
    features.append(len(obstacles) / MAX_OBSTACLES)
    
    # 8. NEW: Physics hints (2 features)
    # Optimal angle estimate (simple physics)
    if targets:
        target = targets[0]
        dx = target.body.position.x - bird.body.position.x
        dy = target.body.position.y - bird.body.position.y
        # Simple angle estimate (ignoring gravity for now)
        optimal_angle = math.atan2(dy, dx) * 180 / math.pi
        features.append(optimal_angle / 90.0)  # Normalized to [-1, 1]
        
        # Distance-based impulse hint
        distance = math.sqrt(dx**2 + dy**2)
        optimal_impulse = min(distance * 2, 2000)  # Simple heuristic
        features.append(optimal_impulse / 2000.0)  # Normalized
    else:
        features.append(0.0)
        features.append(0.5)
    
    # Calculate total expected features
    base_features = 2 + 2 + 4  # bird + target + relative
    obstacle_features = MAX_OBSTACLES * 4
    new_features = 3 + 2 + 2 + 2  # density + target-obstacle + game + physics
    total_expected = base_features + obstacle_features + new_features
    
    # Pad to fixed size
    features = features[:total_expected]
    features = features + [0] * (total_expected - len(features))
    
    return torch.tensor(features, dtype=torch.float32)

def get_action(policy, state):
    feature_vec = flatten_normalize_state(state)
    
    max_angle = 85
    max_impulse_magnitude = 2000
    min_impulse_magnitude = 100
    
    # Get policy distribution
    mean_a, log_std_a, mean_f, log_std_f = policy(feature_vec)
    
    # Clamp log_std to get reasonable std ranges
    log_std_a = torch.clamp(log_std_a, -2, 1)  # std_a will be [0.14, 2.7]
    log_std_f = torch.clamp(log_std_f, -2, 3)  # std_f will be [0.14, 20.1]
    
    std_a = log_std_a.exp()
    std_f = log_std_f.exp()
    
    # Check for NaN values and replace with safe defaults
    if torch.isnan(mean_a) or torch.isnan(std_a):
        mean_a = torch.tensor(0.0)
        std_a = torch.tensor(1.0)
    if torch.isnan(mean_f) or torch.isnan(std_f):
        mean_f = torch.tensor(0.0)
        std_f = torch.tensor(1.0)

    # Map network outputs to action space
    mean_a_scaled = torch.sigmoid(mean_a) * max_angle
    mean_f_scaled = torch.sigmoid(mean_f) * (max_impulse_magnitude - min_impulse_magnitude) + min_impulse_magnitude
    
    # Use std values directly (no second clamping)
    std_a_scaled = std_a
    std_f_scaled = std_f
    
    # Create distributions in scaled space
    dist_a = torch.distributions.Normal(mean_a_scaled, std_a_scaled)
    dist_f = torch.distributions.Normal(mean_f_scaled, std_f_scaled)

    # Sample actions directly in scaled space
    sampled_angle_deg_scaled = dist_a.sample()
    sampled_impulse_magnitude_scaled = dist_f.sample()
    
    # Clamp to valid ranges (safety check)
    sampled_angle_deg_scaled = torch.clamp(sampled_angle_deg_scaled, 0, max_angle)
    sampled_impulse_magnitude_scaled = torch.clamp(sampled_impulse_magnitude_scaled, min_impulse_magnitude, max_impulse_magnitude)
    
    return {
        "samples": [sampled_angle_deg_scaled, sampled_impulse_magnitude_scaled],
        "distributions": [dist_a, dist_f],
        "mean_std_scaled": [mean_a_scaled, std_a_scaled, mean_f_scaled, std_f_scaled],
        "feature_vec": feature_vec
    }


def monte_carlo_simulation(space, target, max_samples=1000, max_hits=10):
    """
    Run Monte Carlo simulation to find ground truth angles and impulses for hits.
    
    Args:
        space: The physics space with obstacles
        target: The target to hit
        max_samples: Maximum number of samples to try
        max_hits: Maximum number of hits to collect
    
    Returns:
        List of tuples (angle, impulse) for successful hits
    """
    hits = []
    samples = 0
    
    print(f"Running Monte Carlo simulation (max {max_samples} samples, collecting up to {max_hits} hits)...")
    
    while len(hits) < max_hits and samples < max_samples:
        # Sample random angle and impulse
        angle = random.uniform(0, 85)  # 0 to 85 degrees
        impulse = random.uniform(100, 2000)  # 100 to 2000 impulse magnitude
        
        # Simulate the shot
        hit, distance_to_target = _simulate_shot(space, target, angle, impulse)
        
        samples += 1
        
        if hit:
            hits.append((angle, impulse))
            print(f"Hit {len(hits)}/{max_hits}: angle={angle:.2f}°, impulse={impulse:.2f}")
    
    print(f"Monte Carlo simulation complete: {len(hits)} hits found in {samples} samples")
    return hits


def update_training_plot(fig, axes, episodes, rewards, losses, angles_scaled, impulses_scaled, 
                        mean_angles, std_angles, mean_impulses, std_impulses, distances_to_target, hits):
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
    axes[0, 1].scatter(episodes, losses, alpha=0.6, s=10, c='r')
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
    
    # Plot 7: Average Distance to Target (100 & 1000 Episodes)
    if len(distances_to_target) >= 100:
        # Calculate rolling average of distances for past 100 episodes
        avg_distances_100 = []
        for i in range(100, len(distances_to_target) + 1):
            avg_distances_100.append(np.mean(distances_to_target[i-100:i]))
        avg_episodes_100 = episodes[99:]  # Episodes corresponding to the 100-episode averages
        axes[2, 0].plot(avg_episodes_100, avg_distances_100, 'purple', alpha=0.7, linewidth=1, label='100 episodes')
    
    if len(distances_to_target) >= 1000:
        # Calculate rolling average of distances for past 1000 episodes
        avg_distances_1000 = []
        for i in range(1000, len(distances_to_target) + 1):
            avg_distances_1000.append(np.mean(distances_to_target[i-1000:i]))
        avg_episodes_1000 = episodes[999:]  # Episodes corresponding to the 1000-episode averages
        axes[2, 0].plot(avg_episodes_1000, avg_distances_1000, 'darkviolet', alpha=0.9, linewidth=2, label='1000 episodes')
    
    axes[2, 0].set_title('Average Distance to Target')
    axes[2, 0].set_xlabel('Episode')
    axes[2, 0].set_ylabel('Average Distance')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 8: Hit Rate (100 & 1000 Episodes)
    if len(hits) >= 100:
        # Calculate rolling hit rate for past 100 episodes
        hit_rates_100 = []
        for i in range(100, len(hits) + 1):
            hit_rates_100.append(np.mean(hits[i-100:i]) * 100)  # Convert to percentage
        hit_episodes_100 = episodes[99:]  # Episodes corresponding to the 100-episode hit rates
        axes[2, 1].plot(hit_episodes_100, hit_rates_100, 'g-', alpha=0.7, linewidth=1, label='100 episodes')
    
    if len(hits) >= 1000:
        # Calculate rolling hit rate for past 1000 episodes
        hit_rates_1000 = []
        for i in range(1000, len(hits) + 1):
            hit_rates_1000.append(np.mean(hits[i-1000:i]) * 100)  # Convert to percentage
        hit_episodes_1000 = episodes[999:]  # Episodes corresponding to the 1000-episode hit rates
        axes[2, 1].plot(hit_episodes_1000, hit_rates_1000, 'darkgreen', alpha=0.9, linewidth=2, label='1000 episodes')
    
    axes[2, 1].set_title('Hit Rate')
    axes[2, 1].set_xlabel('Episode')
    axes[2, 1].set_ylabel('Hit Rate (%)')
    axes[2, 1].set_ylim(0, 100)  # Hit rate is between 0% and 100%
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 9: Average Reward (100 & 1000 Episodes)
    if len(rewards) >= 100:
        # Calculate rolling average of rewards for past 100 episodes
        avg_rewards_100 = []
        for i in range(100, len(rewards) + 1):
            avg_rewards_100.append(np.mean(rewards[i-100:i]))
        avg_episodes_100 = episodes[99:]  # Episodes corresponding to the 100-episode averages
        axes[2, 2].plot(avg_episodes_100, avg_rewards_100, 'b-', alpha=0.7, linewidth=1, label='100 episodes')
    
    if len(rewards) >= 1000:
        # Calculate rolling average of rewards for past 1000 episodes
        avg_rewards_1000 = []
        for i in range(1000, len(rewards) + 1):
            avg_rewards_1000.append(np.mean(rewards[i-1000:i]))
        avg_episodes_1000 = episodes[999:]  # Episodes corresponding to the 1000-episode averages
        axes[2, 2].plot(avg_episodes_1000, avg_rewards_1000, 'darkblue', alpha=0.9, linewidth=2, label='1000 episodes')
    
    axes[2, 2].set_title('Average Reward')
    axes[2, 2].set_xlabel('Episode')
    axes[2, 2].set_ylabel('Average Reward')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    # Update the main title
    fig.suptitle(f'RL Training Metrics - Episode {episodes[-1]}', fontsize=16)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)  # Very brief pause to allow plot to update

# save and load the model

def save_model(policy, path):
    torch.save(policy.state_dict(), path)

def load_model(path):
    policy = RLPolicy()
    policy.load_state_dict(torch.load(path))
    return policy

#  train the RL AI for a given number of steps, and return the trained model
def train_rl_ai(num_episodes: int, plot_interval: int = 100, curriculum_switch: int = 5000):
    """Train the RL AI for a given number of steps"""  
    policy = RLPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    
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
    distances_to_target = []
    hits = []  # Add hit tracking
    sum_rewards_history = []
    
    # Initialize the plot with 3x3 subplots
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(3, 3, figsize=(18, 9))
    fig.suptitle('RL Training Metrics - Starting...', fontsize=16)
    plt.tight_layout()
    plt.show(block=False)
    
    sum_reward = 0
    for episode_idx in range(num_episodes):
        state = initialize_env(MAX_OBSTACLES=5)
        actions_dict = get_action(policy, state)

        angle_deg_scaled, impulse_magnitude_scaled = actions_dict["samples"]
        dist_a, dist_f = actions_dict["distributions"]
        feature_vec = actions_dict["feature_vec"]
        mean_angles_scaled, std_angles_scaled, mean_impulses_scaled, std_impulses_scaled = actions_dict["mean_std_scaled"]

        space, bird, targets, obstacles = state
        target = targets[0]

        reward, distance_to_target = run_env_simulation(space, target, float(angle_deg_scaled.item()), float(impulse_magnitude_scaled.item()))
        
        # Track if this was a hit
        hit = 1 if reward > 0 else 0
        
        if episode_idx < curriculum_switch:
            if reward > 0:
                reward = reward
            else:
                reward = -distance_to_target   
        else:
            reward = reward
        reward = torch.tensor(reward, dtype=torch.float32)

        # Compute policy loss (REINFORCE)
        log_prob = dist_a.log_prob(angle_deg_scaled) + dist_f.log_prob(impulse_magnitude_scaled)
        
        # Add entropy regularization to encourage exploration

        entropy_coefficient = 0.01 
        entropy_loss = -(dist_a.entropy().mean() + dist_f.entropy().mean())
        
        loss = -log_prob * reward - entropy_coefficient * entropy_loss
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
        distances_to_target.append(distance_to_target)
        hits.append(hit)  # Track hit
        sum_rewards_history.append(sum_reward.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step() # Step the scheduler

        # Calculate gradient norms for console output only
        grad_norms = []
        for param in policy.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
            else:
                grad_norms.append(0.0)
        total_grad_norm = sum(grad_norms)

        if episode_idx % 100 == 0:
            print(f'--------------------------------')
            print(f"Episode {episode_idx}, reward={reward}, loss={loss}, log_prob={log_prob} , distance_to_target={distance_to_target}, sum_reward={sum_reward}")
            print(f'Action taken: angle_deg_scaled: {angle_deg_scaled}, impulse_magnitude_scaled: {impulse_magnitude_scaled}')
            print(f'Policy mean and std: mean_a: {dist_a.mean}, std_a: {dist_a.scale}, mean_f: {dist_f.mean}, std_f: {dist_f.scale}')
            print(f'Policy mean and std scaled: mean_angles_scaled: {mean_angles_scaled}, std_angles_scaled: {std_angles_scaled}, mean_impulses_scaled: {mean_impulses_scaled}, std_impulses_scaled: {std_impulses_scaled}')
            print(f'target: {[target.body.position.x, target.body.position.y]}')
            print(f'obstacles: {[[obs.body.position.x, obs.body.position.y] for obs, size in obstacles]}')
            print(f'total gradient norm: {total_grad_norm:.6f}')
            print(f'feature_vec: {feature_vec}')
            
            # Run Monte Carlo simulation every 100 episodes
            print(f"\n=== Monte Carlo Ground Truth Simulation (Episode {episode_idx}) ===")
            monte_carlo_hits = monte_carlo_simulation(space, target, max_samples=1000, max_hits=10)
            
            sum_reward = 0
            
        # Update plot at specified intervals
        if episode_idx % plot_interval == 0:
            update_training_plot(fig, axes, episodes, rewards, losses, angles_scaled, impulses_scaled,
                               mean_angles, std_angles, mean_impulses, std_impulses, distances_to_target, hits)
    
    # Keep the plot open at the end
    # plt.ioff()  # Turn off interactive mode
    # plt.show()

    # save screenshot of the plot
    plt.savefig("rl_training_plot.png")
    
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
    policy = train_rl_ai(30000, plot_interval=100, curriculum_switch=20000)
    save_model(policy, "rl_policy.pth")

# 30% hit rate at 10000 episodes with curriculum switch. entropy coefficient 0.1. tanh scaling. lr scheduler starting at 0.0001.
