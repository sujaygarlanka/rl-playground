import torch
from torch.distributions import Categorical
import os

def test_model(policy, env, num_episodes=10, render=False):
    """
    Test the model and return average reward and success rate
    
    Args:
        policy: The policy network to test
        env: The environment to test in
        num_episodes: Number of episodes to run for testing
        render: Whether to render the environment during testing
    
    Returns:
        avg_reward: Average reward across all test episodes
        success_rate: Percentage of episodes that reach 500 steps
        total_rewards: List of rewards for each episode
    """
    policy.eval()  # Set to evaluation mode
    total_rewards = []
    successful_episodes = 0
    
    for episode in range(num_episodes):
        state, info = env.reset()  # Fixed: unpack both observation and info
        episode_reward = 0
        episode_steps = 0
        
        done = False
        while not done:
            if render:
                env.render()
                
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action_probs = policy(state_tensor)
                dist = Categorical(action_probs)
                action = dist.sample()
            
            # Fixed: unpack all 5 return values from step()
            state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated  # Check both termination conditions
            episode_reward += reward
            episode_steps += 1
            
            # Check if episode was successful (500 steps = solved)
            if episode_steps >= 500:
                successful_episodes += 1
        
        total_rewards.append(episode_reward)
    
    policy.train()  # Set back to training mode
    avg_reward = sum(total_rewards) / num_episodes
    success_rate = (successful_episodes / num_episodes) * 100
    
    return avg_reward, success_rate, total_rewards

def evaluate_and_save_model(policy, optimizer, test_env, episode, train_reward, 
                          best_test_reward, best_test_success_rate, writer, 
                          models_dir='models', model_name='reinforce_cartpole', 
                          num_test_episodes=10):
    """
    Evaluate the model and save if performance improves
    
    Args:
        policy: The policy network
        optimizer: The optimizer
        test_env: Environment for testing
        episode: Current episode number
        train_reward: Training episode reward
        best_test_reward: Best test reward so far
        best_test_success_rate: Best test success rate so far
        writer: TensorBoard writer
        models_dir: Directory to save models
        model_name: Base name for model files
        num_test_episodes: Number of episodes for testing
    
    Returns:
        best_test_reward: Updated best test reward
        best_test_success_rate: Updated best test success rate
        model_saved: Whether a model was saved
    """
    # Test the model
    test_reward, test_success_rate, test_rewards = test_model(policy, test_env, num_episodes=num_test_episodes)
    
    # Log test metrics to TensorBoard
    writer.add_scalar('Test/Average_Reward', test_reward, episode)
    writer.add_scalar('Test/Success_Rate', test_success_rate, episode)
    writer.add_scalar('Test/Reward_Std', torch.tensor(test_rewards).std().item(), episode)
    
    # Check if this is the best performance so far
    improved = False
    if test_reward > best_test_reward:
        best_test_reward = test_reward
        improved = True
        
    if test_success_rate > best_test_success_rate:
        best_test_success_rate = test_success_rate
        improved = True
    
    # Save model if performance improved
    model_saved = False
    if improved:
        model_path = save_model(
            model=policy,
            optimizer=optimizer,
            episode=episode,
            test_reward=test_reward,
            test_success_rate=test_success_rate,
            models_dir=models_dir,
            model_name=model_name
        )
        print(f"Model saved! Episode {episode}: Test Reward={test_reward:.1f}, Success Rate={test_success_rate:.1f}%")
        model_saved = True
    
    print(f"Episode {episode}: Train Reward={train_reward}, Test Reward={test_reward:.1f}, "
          f"Test Success Rate={test_success_rate:.1f}%")
    
    return best_test_reward, best_test_success_rate, model_saved

def save_model(model, optimizer, episode, test_reward, test_success_rate, 
               models_dir='models', model_name='reinforce_cartpole'):
    """
    Save model checkpoint with performance metrics
    
    Args:
        model: The model to save
        optimizer: The optimizer to save
        episode: Current episode number
        test_reward: Current test reward
        test_success_rate: Current test success rate
        models_dir: Directory to save models in
        model_name: Base name for the model file
    """
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(
        models_dir, 
        f'{model_name}_episode_{episode}_reward_{test_reward:.1f}_success_{test_success_rate:.1f}.pth'
    )
    
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_reward': test_reward,
        'test_success_rate': test_success_rate,
    }, model_path)
    
    return model_path

def load_model(model, optimizer, model_path):
    """
    Load model checkpoint
    
    Args:
        model: The model to load state into
        optimizer: The optimizer to load state into
        model_path: Path to the checkpoint file
    
    Returns:
        checkpoint: Dictionary containing checkpoint data
    """
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint