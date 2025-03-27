import numpy as np
import torch
from .weather import WeatherSimulation


class SVRPEnvironment:
    """
    Environment for the Stochastic Vehicle Routing Problem.
    Handles state transitions, rewards, and stochastic variables.
    """
    
    def __init__(self, 
                 num_nodes, 
                 num_vehicles, 
                 capacity, 
                 weather_dim=3,
                 a_ratio=0.6,
                 b_ratio=0.2,
                 gamma_ratio=0.2,
                 device='cpu'):
        """
        Args:
            num_nodes: Number of nodes (customers + depot)
            num_vehicles: Number of vehicles
            capacity: Maximum capacity of each vehicle
            weather_dim: Dimension of weather variables
            a_ratio: Constant component ratio of stochastic variables
            b_ratio: Weather component ratio of stochastic variables
            gamma_ratio: Noise component ratio of stochastic variables
            device: Device to use for tensor operations
        """
        self.num_nodes = num_nodes
        self.num_vehicles = num_vehicles
        self.capacity = capacity
        self.weather_dim = weather_dim
        self.a_ratio = a_ratio
        self.b_ratio = b_ratio
        self.gamma_ratio = gamma_ratio
        self.device = device
        
        # Weather simulation
        self.weather_sim = WeatherSimulation(
            weather_dim=weather_dim,
            a_ratio=a_ratio,
            b_ratio=b_ratio,
            gamma_ratio=gamma_ratio
        )
        
        # Initialize state dimensions
        self.customer_features_dim = weather_dim + 1 + num_nodes  # weather + demand + travel costs
        self.vehicle_features_dim = 2  # position (one-hot) + load
        
    def reset(self, batch_size=1, fixed_customers=True):
        """
        Reset the environment for a new episode.
        
        Args:
            batch_size: Number of parallel environments
            fixed_customers: If True, use fixed customer positions
            
        Returns:
            customer_features: Customer information tensor
            vehicle_features: Vehicle information tensor
            demands: Remaining demand tensor
        """
        # Generate weather, demands, and travel costs
        self.weather, self.demands, self.travel_costs = self.weather_sim.generate(
            batch_size=batch_size,
            num_nodes=self.num_nodes,
            fixed_customers=fixed_customers
        )
        
        # Initial demands
        self.remaining_demands = self.demands.clone()
        
        # Initial vehicle states
        self.vehicle_positions = torch.zeros(
            batch_size, self.num_vehicles, dtype=torch.long, device=self.device
        )  # All vehicles start at depot (node 0)
        
        self.vehicle_loads = torch.full(
            (batch_size, self.num_vehicles), 
            self.capacity, 
            dtype=torch.float32,
            device=self.device
        )  # All vehicles start with full capacity
        
        # Initialize step counter
        self.steps = 0
        
        # Get initial features
        customer_features, vehicle_features = self._get_features()
        
        return customer_features, vehicle_features, self.remaining_demands
    
    def step(self, actions):
        """
        Execute actions and update environment state.
        
        Args:
            actions: Tensor of shape [batch_size, num_vehicles]
                   containing node indices to visit
            
        Returns:
            customer_features: Updated customer information
            vehicle_features: Updated vehicle information
            demands: Updated remaining demand
            rewards: Negative travel costs
            done: Whether episode is complete
        """
        batch_size = actions.size(0)
        rewards = torch.zeros(batch_size, device=self.device)
        
        # For each vehicle, compute travel costs and update states
        for v in range(self.num_vehicles):
            # Get current positions and next positions
            current_positions = self.vehicle_positions[:, v]
            next_positions = actions[:, v]
            
            # Compute travel costs
            for b in range(batch_size):
                current = current_positions[b].item()
                next_node = next_positions[b].item()
                
                # Add travel cost
                rewards[b] -= self.travel_costs[b, current, next_node]
                
                # Update vehicle load and remaining demand
                if next_node > 0:  # Not depot
                    # Calculate how much can be delivered
                    delivery = torch.min(
                        self.vehicle_loads[b, v],
                        self.remaining_demands[b, next_node]
                    )
                    
                    # Update vehicle load and remaining demand
                    self.vehicle_loads[b, v] -= delivery
                    self.remaining_demands[b, next_node] -= delivery
                    
                    # If vehicle cannot fulfill demand, record failure
                    if self.remaining_demands[b, next_node] > 0 and self.vehicle_loads[b, v] <= 0:
                        # Vehicle needs to return to depot for refill
                        # Add recourse cost (depot to customer and back)
                        rewards[b] -= 2 * self.travel_costs[b, 0, next_node]
                        
                        # Refill vehicle
                        self.vehicle_loads[b, v] = self.capacity
                else:
                    # Refill at depot
                    self.vehicle_loads[b, v] = self.capacity
            
            # Update vehicle positions
            self.vehicle_positions[:, v] = next_positions
        
        # Update step counter
        self.steps += 1
        
        # Check if done (all demands fulfilled)
        done = (self.remaining_demands[:, 1:].sum(dim=1) <= 0)
        
        # Get updated features
        customer_features, vehicle_features = self._get_features()
        
        return customer_features, vehicle_features, self.remaining_demands, rewards, done
    
    def _get_features(self):
        """
        Construct feature tensors for the current state.
        
        Returns:
            customer_features: Tensor with customer information
            vehicle_features: Tensor with vehicle information
        """
        batch_size = self.weather.size(0)
        
        # Customer features: weather + demand + travel costs
        customer_features = torch.zeros(
            batch_size, self.num_nodes, self.customer_features_dim, 
            device=self.device
        )
        
        # Add weather to all nodes
        weather_expanded = self.weather.unsqueeze(1).expand(-1, self.num_nodes, -1)
        customer_features[:, :, :self.weather_dim] = weather_expanded
        
        # Add remaining demand
        customer_features[:, :, self.weather_dim] = self.remaining_demands
        
        # Add travel costs (to all other nodes)
        customer_features[:, :, self.weather_dim+1:] = self.travel_costs
        
        # Vehicle features: position (one-hot) + load
        vehicle_features = torch.zeros(
            batch_size, self.num_vehicles, self.vehicle_features_dim,
            device=self.device
        )
        
        # Add position as one-hot (simplified as just the node index for now)
        vehicle_features[:, :, 0] = self.vehicle_positions.float()
        
        # Add load
        vehicle_features[:, :, 1] = self.vehicle_loads / self.capacity  # Normalize load
        
        return customer_features, vehicle_features