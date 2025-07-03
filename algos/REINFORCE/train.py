import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import datetime
from utils import evaluate_and_save_model, test_model
import os

# Create TensorBoard writer
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, f'runs/reinforce_cartpole_{current_time}')
models_dir = os.path.join(script_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.fc(state)
    
def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns
 
# Setup environment and model
env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")  # Separate environment for testing
policy = PolicyNetwork(state_dim=4, action_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)

# Metrics tracking
successful_episodes = 0  # Episodes that reach 500 steps (CartPole solved)
total_steps = 0
best_test_reward = 0
best_test_success_rate = 0
test_interval = 50  # Test every 50 episodes

print(f"Training REINFORCE on CartPole...")
print(f"TensorBoard logs saved to: {log_dir}")
print(f"Models will be saved to: {models_dir}")

# Training loop
for episode in range(2000):
    state, info = env.reset()  # Fixed: unpack both observation and info
    log_probs = []
    rewards = []
    episode_steps = 0

    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = policy(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        # Fixed: unpack all 5 return values from step()
        state, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated  # Check both termination conditions
        rewards.append(reward)
        episode_steps += 1
        total_steps += 1

    # Check if episode was successful (500 steps = solved)
    if episode_steps >= 500:
        successful_episodes += 1

    # Compute returns
    returns = compute_returns(rewards)
    returns = torch.tensor(returns)

    # Normalize returns (helps stability)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute loss
    loss = -torch.sum(torch.stack(log_probs) * returns)
    
    # Compute policy entropy for exploration monitoring (across all actions in episode)
    stacked_log_probs = torch.stack(log_probs)
    policy_entropy = -torch.mean(stacked_log_probs)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log metrics to TensorBoard
    episode_reward = sum(rewards)
    success_rate = (successful_episodes / (episode + 1)) * 100
    
    writer.add_scalar('Episode/Reward', episode_reward, episode)
    writer.add_scalar('Episode/Length', episode_steps, episode)
    writer.add_scalar('Episode/Success_Rate', success_rate, episode)
    writer.add_scalar('Training/Loss', loss.item(), episode)
    writer.add_scalar('Training/Policy_Entropy', policy_entropy.item(), episode)
    writer.add_scalar('Training/Total_Steps', total_steps, episode)
    writer.add_scalar('Training/Average_Return', returns.mean().item(), episode)
    writer.add_scalar('Training/Return_Std', returns.std().item(), episode)

    # Test model periodically
    if (episode + 1) % test_interval == 0:
        best_test_reward, best_test_success_rate, _ = evaluate_and_save_model(
            policy=policy,
            optimizer=optimizer,
            test_env=test_env,
            episode=episode + 1,
            train_reward=episode_reward,
            best_test_reward=best_test_reward,
            best_test_success_rate=best_test_success_rate,
            writer=writer,
            models_dir=models_dir,
            model_name='reinforce_cartpole'
        )

    # Print progress every 50 episodes (if not testing)
    elif (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}: Reward={episode_reward}, Steps={episode_steps}, "
              f"Loss={loss.item():.3f}, Success Rate={success_rate:.1f}%")

print(f"Training completed!")
print(f"Final Stats: {successful_episodes} successful episodes out of 10000, {success_rate:.1f}% success rate")
print(f"Best Test Reward: {best_test_reward:.1f}")
print(f"Best Test Success Rate: {best_test_success_rate:.1f}%")

# Final test with rendering
print("\nRunning final test with rendering...")
final_reward, final_success_rate, _ = test_model(policy, test_env, num_episodes=1, render=True)
print(f"Final Test: Reward={final_reward:.1f}, Success Rate={final_success_rate:.1f}%")

env.close()
test_env.close()
writer.close()
print(f"To view TensorBoard: tensorboard --logdir={log_dir}")