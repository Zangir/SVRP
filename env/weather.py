import numpy as np
import torch


class WeatherSimulation:
    """
    Simulates weather variables and their influence on demands and travel costs.
    """
    
    def __init__(self, weather_dim=3, a_ratio=0.6, b_ratio=0.2, gamma_ratio=0.2, seed=None):
        """
        Args:
            weather_dim: Dimension of weather variables (temperature, pressure, humidity)
            a_ratio: Constant component ratio of stochastic variables
            b_ratio: Weather component ratio of stochastic variables
            gamma_ratio: Noise component ratio of stochastic variables
            seed: Random seed for reproducibility
        """
        self.weather_dim = weather_dim
        self.a_ratio = a_ratio
        self.b_ratio = b_ratio
        self.gamma_ratio = gamma_ratio
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Fixed customer positions for deterministic scenarios
        self.fixed_customer_positions = None
        
    def generate(self, batch_size, num_nodes, fixed_customers=True, device='cpu'):
        """
        Generate weather, demands, and travel costs for a batch of scenarios.
        
        Args:
            batch_size: Number of scenarios to generate
            num_nodes: Number of nodes (customers + depot)
            fixed_customers: If True, use fixed customer positions
            device: Device to use for tensor operations
            
        Returns:
            weather: Tensor of shape [batch_size, weather_dim]
            demands: Tensor of shape [batch_size, num_nodes]
            travel_costs: Tensor of shape [batch_size, num_nodes, num_nodes]
        """
        # Generate weather variables
        weather = torch.zeros(batch_size, self.weather_dim, device=device)
        for i in range(self.weather_dim):
            weather[:, i] = torch.distributions.Uniform(-1, 1).sample([batch_size])
        
        # Generate or reuse customer positions
        if fixed_customers and self.fixed_customer_positions is not None:
            customer_positions = self.fixed_customer_positions.repeat(batch_size, 1, 1)
        else:
            # Node 0 is depot at (0.5, 0.5), rest are random positions
            customer_positions = torch.zeros(batch_size, num_nodes, 2, device=device)
            customer_positions[:, 0] = torch.tensor([0.5, 0.5], device=device)  # Depot position
            
            # Generate random positions for customers
            for b in range(batch_size):
                for i in range(1, num_nodes):
                    customer_positions[b, i] = torch.rand(2, device=device)  # Random in [0,1]
            
            # Save positions if fixed_customers
            if fixed_customers:
                self.fixed_customer_positions = customer_positions[0].unsqueeze(0)
        
        # Generate demands (node 0 is depot, has no demand)
        demands = torch.zeros(batch_size, num_nodes, device=device)
        
        # Generate base demands (constant component)
        base_demands = torch.ones(num_nodes, device=device) * 10
        base_demands[0] = 0  # No demand at depot
        
        # Generate demands for each node
        for b in range(batch_size):
            for i in range(1, num_nodes):  # Skip depot
                # Constant component
                constant = base_demands[i] * self.a_ratio
                
                # Weather component (interaction terms)
                weather_effect = 0
                for j in range(self.weather_dim):
                    for k in range(self.weather_dim):
                        # Random coefficient for weather interaction
                        alpha = torch.randn(1, device=device) * 0.5
                        weather_effect += alpha * weather[b, j] * weather[b, k]
                
                weather_effect *= self.b_ratio * base_demands[i]
                
                # Noise component
                noise = torch.randn(1, device=device) * self.gamma_ratio * base_demands[i]
                
                # Combine components
                demands[b, i] = constant + weather_effect + noise
                
                # Ensure demand is positive
                demands[b, i] = torch.max(demands[b, i], torch.tensor(1.0, device=device))
        
        # Generate travel costs
        travel_costs = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        
        # Calculate Euclidean distances between nodes
        for b in range(batch_size):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        # Base cost is Euclidean distance
                        dist = torch.norm(customer_positions[b, i] - customer_positions[b, j])
                        base_cost = dist * 10  # Scale for better numerical properties
                        
                        # Constant component
                        constant = base_cost * self.a_ratio
                        
                        # Weather component
                        weather_effect = 0
                        for k in range(self.weather_dim):
                            for l in range(self.weather_dim):
                                # Random coefficient for weather interaction
                                alpha = torch.randn(1, device=device) * 0.5
                                weather_effect += alpha * weather[b, k] * weather[b, l]
                        
                        weather_effect *= self.b_ratio * base_cost
                        
                        # Noise component
                        noise = torch.randn(1, device=device) * self.gamma_ratio * base_cost
                        
                        # Combine components
                        travel_costs[b, i, j] = constant + weather_effect + noise
                        
                        # Ensure cost is positive
                        travel_costs[b, i, j] = torch.max(travel_costs[b, i, j], 
                                                          torch.tensor(0.1, device=device))
        
        return weather, demands, travel_costs
    
    def generate_dataset(self, num_scenarios, num_nodes, fixed_customers=True, device='cpu'):
        """
        Generate a dataset of scenarios for k-NN estimation.
        
        Args:
            num_scenarios: Number of scenarios to generate
            num_nodes: Number of nodes (customers + depot)
            fixed_customers: If True, use fixed customer positions
            device: Device to use for tensor operations
            
        Returns:
            dataset: Dictionary containing weather, demands, and travel_costs
        """
        weather, demands, travel_costs = self.generate(
            batch_size=num_scenarios,
            num_nodes=num_nodes,
            fixed_customers=fixed_customers,
            device=device
        )
        
        return {
            'weather': weather,
            'demands': demands,
            'travel_costs': travel_costs
        }