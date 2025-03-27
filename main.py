import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import logging
from tqdm import tqdm

from models.policy import SVRPPolicy
from env.svrp_env import SVRPEnvironment
from training.reinforce import ReinforceTrainer
from inference.inference import GreedyInference, RandomSamplingInference, BeamSearchInference


def parse_args():
    parser = argparse.ArgumentParser(description='SVRP-RL')
    
    # Environment settings
    parser.add_argument('--num_nodes', type=int, default=20, help='Number of nodes (customers + depot)')
    parser.add_argument('--num_vehicles', type=int, default=1, help='Number of vehicles')
    parser.add_argument('--capacity', type=float, default=50.0, help='Vehicle capacity')
    parser.add_argument('--a_ratio', type=float, default=0.6, help='Constant component ratio')
    parser.add_argument('--b_ratio', type=float, default=0.2, help='Weather component ratio')
    parser.add_argument('--gamma_ratio', type=float, default=0.2, help='Noise component ratio')
    parser.add_argument('--weather_dim', type=int, default=3, help='Weather dimension')
    parser.add_argument('--fixed_customers', action='store_true', help='Use fixed customer positions')
    
    # Model settings
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--baseline_lr', type=float, default=1e-3, help='Baseline learning rate')
    parser.add_argument('--entropy_weight', type=float, default=0.01, help='Entropy regularization weight')
    parser.add_argument('--max_steps', type=int, default=100, help='Maximum steps per episode')
    
    # Inference settings
    parser.add_argument('--inference', type=str, default='beam', choices=['greedy', 'random', 'beam'], help='Inference strategy')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples for random sampling')
    parser.add_argument('--beam_width', type=int, default=3, help='Beam width for beam search')
    parser.add_argument('--test_size', type=int, default=100, help='Number of test instances')
    
    # Other settings
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--load_model', type=str, default=None, help='Path to load model from')
    parser.add_argument('--test', action='store_true', help='Test mode (no training)')
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval for training')
    parser.add_argument('--save_interval', type=int, default=20, help='Save interval for models')
    parser.add_argument('--reoptimization', action='store_true', help='Use reoptimization strategy')
    
    return parser.parse_args()


def setup_logger(save_dir):
    """Set up the logger for training and testing."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    log_file = os.path.join(save_dir, 'svrp_log.txt')
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger()


def visualize_route(env, routes, title=None, save_path=None):
    """
    Visualize the routes.
    
    Args:
        env: SVRP environment
        routes: List of routes for each vehicle
        title: Title for the plot
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 8))
    
    # Get customer positions from the weather simulation
    # This assumes env.weather_sim.fixed_customer_positions exists
    if hasattr(env.weather_sim, 'fixed_customer_positions') and env.weather_sim.fixed_customer_positions is not None:
        customer_positions = env.weather_sim.fixed_customer_positions[0].cpu().numpy()
    else:
        # If fixed customer positions aren't available, generate random positions for visualization
        # This is just a fallback for visualization
        num_nodes = env.num_nodes
        customer_positions = np.zeros((num_nodes, 2))
        customer_positions[0] = [0.5, 0.5]  # Depot at center
        for i in range(1, num_nodes):
            customer_positions[i] = np.random.rand(2)
        print("Warning: Using random positions for visualization")
    
    # Plot customer positions
    plt.scatter(customer_positions[1:, 0], customer_positions[1:, 1], c='blue', s=50, label='Customers')
    plt.scatter(customer_positions[0, 0], customer_positions[0, 1], c='red', s=100, marker='*', label='Depot')
    
    # Plot routes
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for v, route in enumerate(routes):
        color = colors[v % len(colors)]
        if not route:
            continue
            
        # Convert route to positions
        positions = []
        positions.append(customer_positions[0])  # Start at depot
        
        for node in route:
            if node < len(customer_positions):
                positions.append(customer_positions[node])
        
        positions.append(customer_positions[0])  # Return to depot
        positions = np.array(positions)
        
        # Plot route
        plt.plot(positions[:, 0], positions[:, 1], c=color, linewidth=2, label=f'Vehicle {v+1}')
    
    plt.grid(True)
    plt.legend()
    
    if title:
        plt.title(title)
        
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()


def train(args, env, trainer, logger):
    """
    Train the model.
    
    Args:
        args: Command line arguments
        env: SVRP environment
        trainer: REINFORCE trainer
        logger: Logger
    """
    logger.info(f"Starting training for {args.epochs} epochs")
    
    # Track metrics
    rewards_history = []
    policy_losses = []
    baseline_losses = []
    
    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train for one episode
        reward, policy_loss, baseline_loss = trainer.train_episode(
            env=env,
            batch_size=args.batch_size,
            max_steps=args.max_steps
        )
        
        # Track metrics
        rewards_history.append(reward)
        policy_losses.append(policy_loss)
        baseline_losses.append(baseline_loss)
        
        # Log progress
        if (epoch + 1) % args.log_interval == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} | "
                        f"Reward: {reward:.4f} | "
                        f"Policy Loss: {policy_loss:.4f} | "
                        f"Baseline Loss: {baseline_loss:.4f} | "
                        f"Time: {time.time() - start_time:.2f}s")
        
        # Save model
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}")
            trainer.save_models(save_path)
            logger.info(f"Saved model to {save_path}")
            
            # Evaluate on test set
            test_reward = evaluate(args, env, trainer.policy_model, 10, logger)
            logger.info(f"Test reward: {test_reward:.4f}")
    
    # Save final model
    save_path = os.path.join(args.save_dir, "model_final")
    trainer.save_models(save_path)
    logger.info(f"Saved final model to {save_path}")
    
    # Plot training metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(rewards_history)
    plt.title('Rewards')
    plt.xlabel('Episode')
    
    plt.subplot(1, 3, 2)
    plt.plot(policy_losses)
    plt.title('Policy Losses')
    plt.xlabel('Episode')
    
    plt.subplot(1, 3, 3)
    plt.plot(baseline_losses)
    plt.title('Baseline Losses')
    plt.xlabel('Episode')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))
    plt.close()
    
    logger.info("Training completed")


def evaluate(args, env, policy_model, num_instances=100, logger=None):
    """
    Evaluate the model on multiple instances.
    """
    total_reward = 0.0
    
    # Create inference strategy
    if args.inference == 'greedy':
        inference_strategy = GreedyInference(policy_model, device=next(policy_model.parameters()).device)
    elif args.inference == 'random':
        inference_strategy = RandomSamplingInference(policy_model, device=next(policy_model.parameters()).device)
    else:  # beam
        inference_strategy = BeamSearchInference(policy_model, device=next(policy_model.parameters()).device)
    
    # Evaluate on multiple instances
    for i in tqdm(range(num_instances), desc="Evaluating"):
        # Solve instance with appropriate parameters for each strategy
        if args.inference == 'random':
            routes, cost = inference_strategy.solve(env=env, num_samples=args.num_samples)
        elif args.inference == 'beam':
            routes, cost = inference_strategy.solve(env=env, beam_width=args.beam_width)
        else:  # greedy
            routes, cost = inference_strategy.solve(env=env)
        
        total_reward -= cost  # Reward is negative cost
        
        # Log instance results
        if logger and i < 5:  # Only log first 5 instances
            logger.info(f"Instance {i+1}: Cost = {cost:.4f}, Routes = {routes}")
            
            # Visualize route for first instance
            if i == 0:
                visualize_route(
                    env=env,
                    routes=routes,
                    title=f"Routes (Cost: {cost:.4f})",
                    save_path=os.path.join(args.save_dir, f"route_{args.inference}.png")
                )
    
    # Calculate mean reward
    mean_reward = total_reward / num_instances
    
    if logger:
        logger.info(f"Evaluation completed. Mean reward: {mean_reward:.4f}")
    
    return mean_reward


def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Set up logger
    logger = setup_logger(args.save_dir)
    logger.info(f"Starting SVRP-RL with {args}")
    logger.info(f"Using device: {device}")
    
    # Create environment
    env = SVRPEnvironment(
        num_nodes=args.num_nodes,
        num_vehicles=args.num_vehicles,
        capacity=args.capacity,
        weather_dim=args.weather_dim,
        a_ratio=args.a_ratio,
        b_ratio=args.b_ratio,
        gamma_ratio=args.gamma_ratio,
        device=device
    )
    
    # Determine input dimensions
    customer_input_dim = env.customer_features_dim
    vehicle_input_dim = env.vehicle_features_dim
    
    logger.info(f"Customer features dimension: {customer_input_dim}")
    logger.info(f"Vehicle features dimension: {vehicle_input_dim}")
    
    # Create policy model
    policy_model = SVRPPolicy(
        customer_input_dim=customer_input_dim,
        vehicle_input_dim=vehicle_input_dim,
        embedding_dim=args.embedding_dim
    ).to(device)
    
    # Create trainer
    trainer = ReinforceTrainer(
        policy_model=policy_model,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        baseline_lr=args.baseline_lr,
        entropy_weight=args.entropy_weight,
        device=device
    )
    
    # Load model if specified
    if args.load_model is not None:
        trainer.load_models(args.load_model)
        logger.info(f"Loaded model from {args.load_model}")
    
    # Training or testing
    if not args.test:
        # Train model
        train(args, env, trainer, logger)
    
    # Evaluate model
    logger.info(f"Evaluating model with {args.inference} inference strategy")
    mean_reward = evaluate(args, env, policy_model, args.test_size, logger)
    logger.info(f"Final evaluation - Mean reward: {mean_reward:.4f}")


if __name__ == "__main__":
    main()