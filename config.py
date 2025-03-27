"""
Configuration parameters for the SVRP-RL project.
This file contains default configuration that can be overridden
by command line arguments.
"""

# Environment settings
ENV_CONFIG = {
    'num_nodes': 20,           # Number of nodes (customers + depot)
    'num_vehicles': 1,         # Number of vehicles
    'capacity': 50.0,          # Vehicle capacity
    'weather_dim': 3,          # Dimension of weather variables
    'a_ratio': 0.6,            # Constant component ratio
    'b_ratio': 0.2,            # Weather component ratio
    'gamma_ratio': 0.2,        # Noise component ratio
    'fixed_customers': True,   # Use fixed customer positions
}

# Model settings
MODEL_CONFIG = {
    'embedding_dim': 128,      # Dimension of embeddings
}

# Training settings
TRAIN_CONFIG = {
    'epochs': 100,             # Number of training epochs
    'batch_size': 32,          # Batch size
    'lr': 1e-4,                # Learning rate
    'baseline_lr': 1e-3,       # Baseline learning rate
    'entropy_weight': 0.01,    # Entropy regularization weight
    'max_steps': 100,          # Maximum steps per episode
    'log_interval': 10,        # Log interval
    'save_interval': 20,       # Save interval
}

# Inference settings
INFERENCE_CONFIG = {
    'strategy': 'beam',        # Inference strategy: 'greedy', 'random', or 'beam'
    'num_samples': 16,         # Number of samples for random sampling
    'beam_width': 3,           # Beam width for beam search
    'test_size': 100,          # Number of test instances
    'reoptimization': False,   # Use reoptimization strategy
}

# Experiment configurations for different problem sizes
EXPERIMENT_CONFIGS = {
    'small': {
        'env': {'num_nodes': 10, 'num_vehicles': 1},
        'train': {'batch_size': 64, 'epochs': 50}
    },
    'medium': {
        'env': {'num_nodes': 20, 'num_vehicles': 1},
        'train': {'batch_size': 32, 'epochs': 100}
    },
    'large': {
        'env': {'num_nodes': 50, 'num_vehicles': 2},
        'train': {'batch_size': 16, 'epochs': 200}
    },
    'xlarge': {
        'env': {'num_nodes': 100, 'num_vehicles': 3},
        'train': {'batch_size': 8, 'epochs': 300}
    }
}

# Signal ratio configurations
SIGNAL_RATIO_CONFIGS = {
    'high_constant': {'a_ratio': 0.8, 'b_ratio': 0.0, 'gamma_ratio': 0.2},
    'high_weather': {'a_ratio': 0.6, 'b_ratio': 0.4, 'gamma_ratio': 0.0},
    'balanced': {'a_ratio': 0.6, 'b_ratio': 0.2, 'gamma_ratio': 0.2},
    'high_noise': {'a_ratio': 0.4, 'b_ratio': 0.1, 'gamma_ratio': 0.5}
}

# Fill rate configurations
FILL_RATE_CONFIGS = {
    'low': {'capacity': 20.0},
    'medium': {'capacity': 50.0},
    'high': {'capacity': 100.0}
}

# Experiment runners
def get_experiment_config(size='medium'):
    """Get configuration for a specific experiment size."""
    if size not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment size: {size}")
    
    config = {}
    config.update(ENV_CONFIG)
    config.update(MODEL_CONFIG)
    config.update(TRAIN_CONFIG)
    config.update(INFERENCE_CONFIG)
    
    # Update with experiment-specific config
    experiment_config = EXPERIMENT_CONFIGS[size]
    for key, value in experiment_config.items():
        if key in config:
            config[key].update(value)
        else:
            config[key] = value
    
    return config


def get_signal_ratio_config(ratio_type='balanced'):
    """Get configuration for a specific signal ratio experiment."""
    if ratio_type not in SIGNAL_RATIO_CONFIGS:
        raise ValueError(f"Unknown signal ratio type: {ratio_type}")
    
    config = get_experiment_config('medium')
    config.update(SIGNAL_RATIO_CONFIGS[ratio_type])
    
    return config


def get_fill_rate_config(rate='medium'):
    """Get configuration for a specific fill rate experiment."""
    if rate not in FILL_RATE_CONFIGS:
        raise ValueError(f"Unknown fill rate: {rate}")
    
    config = get_experiment_config('medium')
    config.update(FILL_RATE_CONFIGS[rate])
    
    return config