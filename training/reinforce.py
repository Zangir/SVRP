import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .baseline import BaselineModel


class ReinforceTrainer:
    """
    Implements the REINFORCE algorithm for training the SVRP policy network.
    
    Based on the paper formula:
    ∇_θ J^π(Θ) ≈ (1/S)∑^S_s=1 ∑^T_s_t=1 ((C(I^t_s, h^t_k) - b_φ(I^t_s, h^t_k)) · ∇_Θ log π_Θ(a^t_k|I^t_s, h^t_k))
    """
    
    def __init__(self, 
                 policy_model, 
                 embedding_dim,
                 lr=1e-4,
                 baseline_lr=1e-3,
                 entropy_weight=0.01,
                 device='cpu'):
        """
        Args:
            policy_model: Policy network
            embedding_dim: Dimension of embeddings
            lr: Learning rate for policy network
            baseline_lr: Learning rate for baseline network
            entropy_weight: Weight for entropy regularization
            device: Device to use for tensor operations
        """
        self.policy_model = policy_model
        self.device = device
        self.entropy_weight = entropy_weight
        
        # Initialize optimizer for policy
        self.optimizer = optim.Adam(policy_model.parameters(), lr=lr)
        
        # Initialize baseline model
        self.baseline_model = BaselineModel(embedding_dim).to(device)
        self.baseline_optimizer = optim.Adam(self.baseline_model.parameters(), lr=baseline_lr)
        
        # MSE loss for baseline
        self.mse_loss = nn.MSELoss()
        
        # Tracking
        self.episode_rewards = []
        self.baseline_losses = []
        self.policy_losses = []
        
    def train_episode(self, env, batch_size=32, max_steps=100):
        """
        Train for one episode.
        
        Args:
            env: SVRP environment
            batch_size: Number of parallel environments
            max_steps: Maximum number of steps per episode
            
        Returns:
            mean_reward: Mean reward of the episode
            policy_loss: Policy loss
            baseline_loss: Baseline loss
        """
        # Reset environment
        customer_features, vehicle_features, demands = env.reset(batch_size=batch_size)
        
        # Initialize hidden state
        hidden = None
        
        # Initialize trajectory storage
        log_probs_list = []
        entropies_list = []
        rewards_list = []
        baseline_values_list = []
        mask_list = []  # To handle variable-length episodes
        
        # Episode loop
        done_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        all_done = False
        
        for step in range(max_steps):
            # Forward pass through policy network
            log_probs, hidden = self.policy_model(
                customer_features, vehicle_features, demands, hidden
            )
            
            # Sample actions for each vehicle
            actions = []
            batch_log_probs = []
            batch_entropies = []
            
            for v in range(env.num_vehicles):
                vehicle_log_probs = log_probs[:, v, :]
                
                # Calculate entropy (for exploration regularization)
                probs = torch.exp(vehicle_log_probs)
                entropy = -torch.sum(probs * vehicle_log_probs, dim=-1)
                
                # Sample actions
                if not all_done:
                    action = self.policy_model.sample_action(vehicle_log_probs.unsqueeze(1)).squeeze(1)
                    
                    # Get log probability of chosen action
                    batch_idx = torch.arange(batch_size, device=self.device)
                    action_log_probs = vehicle_log_probs[batch_idx, action]
                    
                    actions.append(action)
                    batch_log_probs.append(action_log_probs)
                    batch_entropies.append(entropy)
            
            # Stack actions for all vehicles
            actions = torch.stack(actions, dim=1)
            batch_log_probs = torch.stack(batch_log_probs, dim=1)
            batch_entropies = torch.stack(batch_entropies, dim=1)
            
            # Compute baseline value
            baseline_value = self.baseline_model(customer_features, vehicle_features)
            
            # Execute actions in environment
            next_customer_features, next_vehicle_features, next_demands, rewards, done = env.step(actions)
            
            # Update done mask
            done_mask = done_mask | done
            all_done = done_mask.all().item()
            
            # Append to trajectory storage
            log_probs_list.append(batch_log_probs)
            entropies_list.append(batch_entropies.mean(dim=1))  # Average entropy across vehicles
            rewards_list.append(rewards)
            baseline_values_list.append(baseline_value.squeeze())
            mask_list.append(~done_mask)  # Store the inverse of done_mask
            
            # Update state
            customer_features = next_customer_features
            vehicle_features = next_vehicle_features
            demands = next_demands
            
            # Break if all environments are done
            if all_done:
                break
        
        # Process collected trajectories
        episode_length = len(rewards_list)
        
        # Convert lists to tensors
        log_probs_tensor = torch.stack(log_probs_list)  # [episode_length, batch_size, num_vehicles]
        entropies_tensor = torch.stack(entropies_list)  # [episode_length, batch_size]
        rewards_tensor = torch.stack(rewards_list)      # [episode_length, batch_size]
        baseline_tensor = torch.stack(baseline_values_list)  # [episode_length, batch_size]
        mask_tensor = torch.stack(mask_list)            # [episode_length, batch_size]
        
        # Calculate total returns
        returns = self._compute_returns(rewards_tensor, mask_tensor)
        
        # Calculate advantages (returns - baseline)
        advantages = returns - baseline_tensor
        
        # Calculate policy loss
        policy_loss = self._compute_policy_loss(log_probs_tensor, entropies_tensor, advantages, mask_tensor)
        
        # Calculate baseline loss
        baseline_loss = self._compute_baseline_loss(baseline_tensor, returns, mask_tensor)
        
        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update baseline
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()
        
        # Calculate mean reward
        mean_reward = rewards_tensor.sum(dim=0).mean().item()
        
        # Record metrics
        self.episode_rewards.append(mean_reward)
        self.policy_losses.append(policy_loss.item())
        self.baseline_losses.append(baseline_loss.item())
        
        return mean_reward, policy_loss.item(), baseline_loss.item()
    
    def _compute_returns(self, rewards, mask, gamma=0.99):
        """
        Compute returns (sum of future rewards).
        
        Args:
            rewards: Tensor of shape [episode_length, batch_size]
            mask: Tensor of shape [episode_length, batch_size]
            gamma: Discount factor
            
        Returns:
            returns: Tensor of shape [episode_length, batch_size]
        """
        episode_length, batch_size = rewards.size()
        returns = torch.zeros_like(rewards)
        
        # Calculate returns
        future_return = torch.zeros(batch_size, device=self.device)
        
        for t in reversed(range(episode_length)):
            # Only include future returns for active environments
            future_return = rewards[t] + gamma * future_return * mask[t]
            returns[t] = future_return
            
        return returns
    
    def _compute_policy_loss(self, log_probs, entropies, advantages, mask):
        """
        Compute policy loss using REINFORCE with baseline.
        
        Args:
            log_probs: Tensor of shape [episode_length, batch_size, num_vehicles]
            entropies: Tensor of shape [episode_length, batch_size]
            advantages: Tensor of shape [episode_length, batch_size]
            mask: Tensor of shape [episode_length, batch_size]
            
        Returns:
            policy_loss: Scalar tensor
        """
        episode_length, batch_size, num_vehicles = log_probs.size()
        
        # Reshape log_probs and repeat advantages for each vehicle
        log_probs = log_probs.view(episode_length, batch_size, num_vehicles)
        advantages = advantages.unsqueeze(-1).expand(-1, -1, num_vehicles)
        mask = mask.unsqueeze(-1).expand(-1, -1, num_vehicles)
        
        # Compute policy gradient
        policy_gradient = -log_probs * advantages.detach() * mask
        
        # Sum across vehicles and steps, average across batch
        policy_loss = policy_gradient.sum(dim=2).sum(dim=0).mean()
        
        # Add entropy regularization
        entropy_loss = -self.entropy_weight * (entropies * mask[:, :, 0]).sum(dim=0).mean()
        
        return policy_loss + entropy_loss
    
    def _compute_baseline_loss(self, baseline_values, returns, mask):
        """
        Compute baseline loss (MSE).
        
        Args:
            baseline_values: Tensor of shape [episode_length, batch_size]
            returns: Tensor of shape [episode_length, batch_size]
            mask: Tensor of shape [episode_length, batch_size]
            
        Returns:
            baseline_loss: Scalar tensor
        """
        # Only compute loss for active environments
        baseline_loss = (((baseline_values - returns) ** 2) * mask).sum(dim=0).mean()
        
        return baseline_loss
    
    def save_models(self, path_prefix):
        """
        Save both policy and baseline models.
        
        Args:
            path_prefix: Path prefix for saving models
        """
        torch.save(self.policy_model.state_dict(), f"{path_prefix}_policy.pt")
        torch.save(self.baseline_model.state_dict(), f"{path_prefix}_baseline.pt")
        
    def load_models(self, path_prefix):
        """
        Load both policy and baseline models.
        
        Args:
            path_prefix: Path prefix for loading models
        """
        self.policy_model.load_state_dict(torch.load(f"{path_prefix}_policy.pt"))
        self.baseline_model.load_state_dict(torch.load(f"{path_prefix}_baseline.pt"))