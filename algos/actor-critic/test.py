import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
import time
import os

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

# Load the trained policy
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'models', 'reinforce_cartpole_episode_600_reward_500.0_success_100.0.pth')
checkpoint = torch.load(model_path, map_location=torch.device('mps'))
policy = PolicyNetwork(state_dim=4, action_dim=2)
policy.load_state_dict(checkpoint['model_state_dict'])
policy.eval()  # Set to evaluation mode

# Create the CartPole environment
env = gym.make('CartPole-v1', render_mode='human')

# Reset the environment
observation, info = env.reset()

print("Press Ctrl+C to stop")

try:
    episode_count = 0
    total_reward = 0
    
    while True:
        # Convert observation to tensor
        state_tensor = torch.tensor(observation, dtype=torch.float32)
        
        # Get action from policy (no exploration, just exploitation)
        with torch.no_grad():
            action_probs = policy(state_tensor)
            dist = Categorical(action_probs)
            action = dist.sample()
        
        # Take action in environment
        observation, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward
        
        # Check if episode is done
        if terminated or truncated:
            episode_count += 1
            print(f"Episode {episode_count} finished! Total reward: {total_reward}")
            total_reward = 0
            observation, info = env.reset()
            time.sleep(0.5)  # Small delay to see the reset

except KeyboardInterrupt:
    print(f"\nStopping CartPole... Ran {episode_count} episodes")
finally:
    env.close()
    print("Environment closed.") 