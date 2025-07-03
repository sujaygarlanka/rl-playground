import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys
import os

torch.autograd.set_detect_anomaly(True)

# Add parent directory to path for utils import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import evaluate_and_save_model, test_model

# Create TensorBoard writer
current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
script_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(script_dir, f'runs/actor_critic_cartpole_{current_time}')
models_dir = os.path.join(script_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state):
        return self.fc(state)
    
# Action-value network
class ActionValueNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
    
    def forward(self, state):
        return self.fc(state)
 
# Setup environment and model
env = gym.make("CartPole-v1")
test_env = gym.make("CartPole-v1")  # Separate environment for testing
policy_network = PolicyNetwork(state_dim=4, action_dim=2)
action_value_network = ActionValueNetwork(state_dim=4, action_dim=2)
policy_optimizer = optim.Adam(policy_network.parameters(), lr=1e-2)
action_value_optimizer = optim.Adam(action_value_network.parameters(), lr=1e-2)

# Metrics tracking
successful_episodes = 0  # Episodes that reach 500 steps (CartPole solved)
total_steps = 0
best_test_reward = 0
best_test_success_rate = 0
test_interval = 50  # Test every 50 episodes

print(f"Training Actor-Critic on CartPole...")
print(f"TensorBoard logs saved to: {log_dir}")
print(f"Models will be saved to: {models_dir}")

# Training loop
for episode in range(5000):
    state, info = env.reset()  # Fixed: unpack both observation and info
    rewards = []
    episode_steps = 0

    state_tensor = torch.tensor(state, dtype=torch.float32)
    action_dist = Categorical(policy_network(state_tensor))
    action_tensor = action_dist.sample()

    done = False
    while not done:

        # Fixed: unpack all 5 return values from step()
        next_state, reward, terminated, truncated, info = env.step(action_tensor.item())
        done = terminated or truncated  # Check both termination conditions

        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        next_action_dist = Categorical(policy_network(next_state_tensor))
        next_action_tensor = next_action_dist.sample()

        action_value_tensor = action_value_network(state_tensor)
        action_value = action_value_tensor.gather(0, action_tensor.unsqueeze(-1)).squeeze(0)
        
        next_action_value_tensor = action_value_network(next_state_tensor)
        next_action_value = next_action_value_tensor.gather(0, next_action_tensor.unsqueeze(-1)).squeeze(0)

        action_value_loss = (reward + 0.99 * next_action_value - action_value)

        action_value_optimizer.zero_grad()
        action_value_loss.backward() 
        action_value_optimizer.step()

        policy_optimizer.zero_grad()
        log_prob = Categorical(policy_network(state_tensor)).log_prob(action_tensor)
        policy_loss = -action_value.detach() * log_prob
        policy_loss.backward()
        policy_optimizer.step()

        rewards.append(reward)
        state_tensor = next_state_tensor
        action_tensor = next_action_tensor

        episode_steps += 1
        total_steps += 1

    # Check if episode was successful (500 steps = solved)
    if episode_steps >= 500:
        successful_episodes += 1

    # Log metrics to TensorBoard
    episode_reward = sum(rewards)
    success_rate = (successful_episodes / (episode + 1)) * 100
    
    writer.add_scalar('Episode/Reward', episode_reward, episode)
    writer.add_scalar('Episode/Length', episode_steps, episode)
    writer.add_scalar('Episode/Success_Rate', success_rate, episode)
    writer.add_scalar('Training/Loss', policy_loss.item(), episode)

    # Test model periodically
    if (episode + 1) % test_interval == 0:
        best_test_reward, best_test_success_rate, _ = evaluate_and_save_model(
            policy=policy_network,
            optimizer=policy_optimizer,
            test_env=test_env,
            episode=episode + 1,
            train_reward=episode_reward,
            best_test_reward=best_test_reward,
            best_test_success_rate=best_test_success_rate,
            writer=writer,
            models_dir=models_dir,
            model_name='actor_critic_cartpole'
        )

    # Print progress every 50 episodes (if not testing)
    elif (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}: Reward={episode_reward}, Steps={episode_steps}, "
              f"Loss={policy_loss.item():.3f}, Success Rate={success_rate:.1f}%")

print(f"Training completed!")
print(f"Final Stats: {successful_episodes} successful episodes out of 2000, {success_rate:.1f}% success rate")
print(f"Best Test Reward: {best_test_reward:.1f}")
print(f"Best Test Success Rate: {best_test_success_rate:.1f}%")

# Final test with rendering
print("\nRunning final test with rendering...")
final_reward, final_success_rate, _ = test_model(policy_network, test_env, num_episodes=1, render=True)
print(f"Final Test: Reward={final_reward:.1f}, Success Rate={final_success_rate:.1f}%")

env.close()
test_env.close()
writer.close()
print(f"To view TensorBoard: tensorboard --logdir={log_dir}")