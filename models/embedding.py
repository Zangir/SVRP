import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomerEncoder(nn.Module):
    """
    Encodes customer information using 1D convolution to create state embeddings.
    
    Processes features such as:
    - Weather variables
    - Customer demand
    - Travel costs to other nodes
    """
    
    def __init__(self, input_dim, embedding_dim):
        """
        Args:
            input_dim: Dimension of input features (weather vars + demand + travel costs)
            embedding_dim: Dimension of the output embedding
        """
        super(CustomerEncoder, self).__init__()
        self.conv1d = nn.Conv1d(1, embedding_dim, kernel_size=input_dim, stride=1)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, num_nodes, input_dim]
               containing customer information
               
        Returns:
            Tensor of shape [batch_size, num_nodes, embedding_dim]
        """
        batch_size, num_nodes, input_dim = x.size()
        
        # Reshape for 1D convolution
        x = x.view(batch_size * num_nodes, 1, input_dim)
        
        # Apply convolution
        x = self.conv1d(x)
        
        # Reshape back
        x = x.view(batch_size, num_nodes, -1)
        
        return x


class VehicleEncoder(nn.Module):
    """
    Encodes vehicle information using LSTM to create memory embeddings.
    
    Processes:
    - Current position
    - Current load
    """
    
    def __init__(self, input_dim, embedding_dim):
        """
        Args:
            input_dim: Dimension of input features (position + load)
            embedding_dim: Dimension of the output embedding
        """
        super(VehicleEncoder, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=embedding_dim,
            batch_first=True
        )
        
    def forward(self, x, hidden=None):
        """
        Args:
            x: Tensor of shape [batch_size, num_vehicles, input_dim]
               containing vehicle information
            hidden: Previous hidden state (for sequential processing)
               
        Returns:
            Tensor of shape [batch_size, num_vehicles, embedding_dim]
            and the updated hidden state
        """
        batch_size, num_vehicles, input_dim = x.size()
        
        # Reshape for LSTM
        x = x.view(batch_size * num_vehicles, 1, input_dim)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(1, batch_size * num_vehicles, self.lstm.hidden_size, 
                             device=x.device)
            c0 = torch.zeros(1, batch_size * num_vehicles, self.lstm.hidden_size, 
                             device=x.device)
            hidden = (h0, c0)
        
        # Apply LSTM
        output, hidden = self.lstm(x, hidden)
        
        # Reshape back
        output = output.view(batch_size, num_vehicles, -1)
        
        return output, hidden