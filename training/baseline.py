import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    """
    Baseline model for estimating the expected return of a state.
    Used to reduce variance in the REINFORCE algorithm.
    
    As described in the paper: b_φ(I^t_s, h^t_k) is trained to minimize
    L(φ) = (1/S)∑^S_s ∑^T_s_t=1 ||b_φ(I^t_s, h^t_k) - C(I^t_s, h^t_k)||²
    """
    
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim: Dimension of state embeddings
        """
        super(BaselineModel, self).__init__()
        
        # Define layers
        input_dim = 26  # Match the actual input dimension
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.fc3 = nn.Linear(embedding_dim // 2, 1)
        
        # Activation functions
        self.relu = nn.ReLU()
        
    def forward(self, customer_features, vehicle_features):
        """
        Estimate the expected return for the given state.
        
        Args:
            customer_features: Tensor of shape [batch_size, num_nodes, feature_dim]
            vehicle_features: Tensor of shape [batch_size, num_vehicles, feature_dim]
            
        Returns:
            Tensor of shape [batch_size, 1] containing estimated returns
        """
        batch_size = customer_features.size(0)
        
        # Average customer features
        avg_customer = torch.mean(customer_features, dim=1)
        
        # Average vehicle features
        avg_vehicle = torch.mean(vehicle_features, dim=1)
        
        # Concatenate features
        x = torch.cat([avg_customer, avg_vehicle], dim=1)
        
        # Forward pass through network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for final output
        
        return x