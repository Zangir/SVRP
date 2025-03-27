import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Attention layer that combines state embeddings (customers) and
    memory embeddings (vehicles) to determine node probabilities.
    """
    
    def __init__(self, embedding_dim):
        """
        Args:
            embedding_dim: Dimension of both state and memory embeddings
        """
        super(AttentionLayer, self).__init__()
        
        # Projections for query, key, value
        self.wq = nn.Linear(embedding_dim, embedding_dim)
        self.wk = nn.Linear(embedding_dim, embedding_dim)
        self.wv = nn.Linear(embedding_dim, embedding_dim)
        
        # Final projection
        self.projection = nn.Linear(embedding_dim, 1)
        
        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.tensor(embedding_dim, dtype=torch.float32))
        
    def forward(self, state_embeddings, memory_embeddings, mask=None):
        """
        Args:
            state_embeddings: Tensor of shape [batch_size, num_nodes, embedding_dim]
                             containing customer information
            memory_embeddings: Tensor of shape [batch_size, num_vehicles, embedding_dim]
                             containing vehicle information
            mask: Boolean tensor of shape [batch_size, num_nodes]
                  where True indicates nodes that should be masked
                  
        Returns:
            Tensor of shape [batch_size, num_vehicles, num_nodes]
            containing probabilities for each node being the next position
        """
        batch_size, num_nodes, _ = state_embeddings.size()
        _, num_vehicles, _ = memory_embeddings.size()
        
        # Compute query, key, value
        q = self.wq(memory_embeddings)  # [batch_size, num_vehicles, embedding_dim]
        k = self.wk(state_embeddings)   # [batch_size, num_nodes, embedding_dim]
        v = self.wv(state_embeddings)   # [batch_size, num_nodes, embedding_dim]
        
        # Compute attention scores
        q = q.unsqueeze(2)                  # [batch_size, num_vehicles, 1, embedding_dim]
        k = k.unsqueeze(1)                  # [batch_size, 1, num_nodes, embedding_dim]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = scores.squeeze(2)          # [batch_size, num_vehicles, num_nodes]
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, num_vehicles, -1)
            scores = scores.masked_fill(mask, -1e9)
        
        # Apply softmax to get probabilities
        probs = F.softmax(scores, dim=-1)   # [batch_size, num_vehicles, num_nodes]
        
        return probs


class MaskingLayer(nn.Module):
    """
    Creates a mask for the attention layer to prevent selecting
    nodes that have already been visited or whose demand is fulfilled.
    """
    
    def __init__(self):
        super(MaskingLayer, self).__init__()
    
    def forward(self, demands):
        """
        Args:
            demands: Tensor of shape [batch_size, num_nodes] containing
                    the remaining demand for each node
                    
        Returns:
            Boolean tensor of shape [batch_size, num_nodes] where
            True indicates nodes that should be masked (demand = 0)
        """
        # Mask nodes with zero demand (already fulfilled)
        # Note: We don't mask the depot (node 0)
        mask = (demands <= 0)
        
        # Never mask the depot (node 0)
        if mask.size(1) > 0:
            mask[:, 0] = False
            
        return mask