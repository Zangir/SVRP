import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import CustomerEncoder, VehicleEncoder
from .attention import AttentionLayer, MaskingLayer


class SVRPPolicy(nn.Module):
    """
    Full policy model for Stochastic Vehicle Routing Problem.
    Combines customer encoder, vehicle encoder, and attention layer.
    """
    
    def __init__(self, 
                 customer_input_dim, 
                 vehicle_input_dim, 
                 embedding_dim):
        """
        Args:
            customer_input_dim: Dimension of customer features
                               (weather + demand + travel costs)
            vehicle_input_dim: Dimension of vehicle features
                              (position + load)
            embedding_dim: Dimension of the embeddings
        """
        super(SVRPPolicy, self).__init__()
        
        # Customer encoder
        self.customer_encoder = CustomerEncoder(
            input_dim=customer_input_dim,
            embedding_dim=embedding_dim
        )
        
        # Vehicle encoder
        self.vehicle_encoder = VehicleEncoder(
            input_dim=vehicle_input_dim,
            embedding_dim=embedding_dim
        )
        
        # Attention layer
        self.attention = AttentionLayer(
            embedding_dim=embedding_dim
        )
        
        # Masking layer
        self.masking = MaskingLayer()
        
        # Save dimensions
        self.embedding_dim = embedding_dim
        
    def forward(self, customer_features, vehicle_features, demands, hidden=None):
        """
        Args:
            customer_features: Tensor of shape [batch_size, num_nodes, customer_input_dim]
            vehicle_features: Tensor of shape [batch_size, num_vehicles, vehicle_input_dim]
            demands: Tensor of shape [batch_size, num_nodes] containing
                    the remaining demand for each node
            hidden: Optional previous hidden state for the vehicle encoder
            
        Returns:
            log_probs: Tensor of shape [batch_size, num_vehicles, num_nodes]
                      containing log probabilities for each node
            hidden: Updated hidden state for sequential processing
        """
        # Encode customer information
        state_embeddings = self.customer_encoder(customer_features)
        
        # Encode vehicle information
        memory_embeddings, hidden = self.vehicle_encoder(vehicle_features, hidden)
        
        # Create mask for fulfilled demands
        mask = self.masking(demands)
        
        # Apply attention to get probabilities
        probs = self.attention(state_embeddings, memory_embeddings, mask)
        
        # Return log probabilities for numerical stability
        log_probs = torch.log(probs + 1e-10)  # Add small constant to avoid log(0)
        
        return log_probs, hidden
    
    def sample_action(self, log_probs, greedy=False):
        """
        Sample actions based on log probabilities.
        
        Args:
            log_probs: Tensor of shape [batch_size, num_vehicles, num_nodes]
                      containing log probabilities
            greedy: If True, select the most probable action
            
        Returns:
            actions: Tensor of shape [batch_size, num_vehicles]
                    containing selected node indices
        """
        if greedy:
            # Select most probable action
            actions = torch.argmax(log_probs, dim=-1)
        else:
            # Sample from probability distribution
            probs = torch.exp(log_probs)
            actions = torch.multinomial(probs.view(-1, log_probs.size(-1)), 1)
            actions = actions.view(log_probs.size(0), log_probs.size(1))
            
        return actions