import argparse
import math
import pickle
import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange


class DataBatchLoader:
    """data loader for joint and marginal distributions"""
    def __init__(self, data, batch_size=128, shuffle=False, x_dim=768):
        self.x = data[:, :x_dim]
        self.y = data[:, x_dim:]
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sample_num = len(data)
        self.index_list = list(range(self.sample_num))
        self.total_steps = int(math.ceil(self.sample_num / batch_size))
        self.reset()

    def reset(self):
        self.step = 0
        if self.shuffle:
            random.shuffle(self.index_list)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.step >= self.total_steps:
            raise StopIteration
        
        start = self.step * self.batch_size
        end = min(start + self.batch_size, self.sample_num)
        indices = self.index_list[start:end]
        
        x_batch = self.x[indices]
        y_batch = self.y[indices]
        
        # Create joint and marginal batches
        joint_batch = np.concatenate((x_batch, y_batch), axis=1)
        shuffled_y = y_batch.copy()
        np.random.shuffle(shuffled_y)
        marginal_batch = np.concatenate((x_batch, shuffled_y), axis=1)
        
        self.step += 1
        return joint_batch, marginal_batch


class MutualInformationEstimator(nn.Module):
    """Neural network for estimating mutual information using MINE"""
    def __init__(self, input_size, hidden_size=100):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.network(x)


class MINETrainer:
    """Trains a MINE model to estimate mutual information"""
    def __init__(self, model, optimizer, device, ma_rate=0.01):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.ma_rate = ma_rate
        self.ma_et = 1.0  # Moving average baseline

    def compute_mi(self, joint, marginal):
        """Compute mutual information lower bound"""
        t = self.model(joint)
        et = torch.exp(self.model(marginal))
        return torch.mean(t) - torch.log(torch.mean(et)), t, et

    def train_step(self, joint_batch, marginal_batch):
        """Perform a single training step"""
        joint = torch.FloatTensor(joint_batch).to(self.device)
        marginal = torch.FloatTensor(marginal_batch).to(self.device)
        
        self.optimizer.zero_grad()
        mi_lb, t, et = self.compute_mi(joint, marginal)
        
        # Update moving average
        self.ma_et = (1 - self.ma_rate) * self.ma_et + self.ma_rate * torch.mean(et).detach()
        
        # Compute loss
        loss = -(torch.mean(t) - (1 / self.ma_et) * torch.mean(et))
        loss.backward()
        self.optimizer.step()
        
        return mi_lb.detach().cpu().item()


def train_model(data, model, optimizer, device, config):
    """
    Train MINE model with progress tracking and checkpointing
    
    Args:
        data: Input dataset (numpy array)
        model: MINE model instance
        optimizer: Model optimizer
        device: Target device (cuda/cpu)
        config: Training configuration dictionary
        
    Returns:
        List of MI estimates during training
    """
    trainer = MINETrainer(model, optimizer, device, ma_rate=config['ma_rate'])
    loader = DataBatchLoader(
        data, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        x_dim=config['x_dim']
    )
    
    results = []
    total_iters = config['epochs'] * loader.total_steps
    progress_bar = tqdm(total=total_iters, desc="Training MINE")
    
    for epoch in range(config['epochs']):
        for joint_batch, marginal_batch in loader:
            mi_estimate = trainer.train_step(joint_batch, marginal_batch)
            results.append(mi_estimate)
            
            # Update progress
            current_iter = len(results)
            progress_bar.update(1)
            progress_bar.set_postfix({"MI": f"{mi_estimate:.4f}", "Epoch": epoch+1})
            
            # Checkpointing and logging
            if current_iter % config['log_freq'] == 0:
                if config['verbose']:
                    print(f"\nIter {current_iter}: MI Estimate = {mi_estimate:.4f}")
                
                save_checkpoint({
                    'iteration': current_iter,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'mi_estimates': results,
                    'ma_et': trainer.ma_et
                }, config['save_path'])
                
                with open(f"{config['output_prefix']}_{current_iter}.pkl", 'wb') as f:
                    pickle.dump(results, f)
    
    # Final save
    save_checkpoint({
        'iteration': len(results),
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'mi_estimates': results,
        'ma_et': trainer.ma_et
    }, config['save_path'])
    
    return results


def save_checkpoint(state, path):
    """Save training checkpoint"""
    torch.save(state, path)
    print(f"Checkpoint saved at iteration {state['iteration']}")


def moving_average(data, window_size=100):
    """Compute moving average of data"""
    return [np.mean(data[i:i+window_size]) for i in range(len(data)-window_size)]


def load_data(emb_path, rating_path):
    """
    Load  train set review embeddings and ratings 
    
    Returns:
        Tuple of (combined data array, x_dimension, y_dimension)
    """
    # Load embeddings and ratings
    embeddings = np.load(emb_path)
    ratings = pd.concat([
        pd.read_csv(f"{rating_path}/train.csv")['rate']
    ]).values
    
    # Convert ratings to one-hot encoding
    y_dim = int(ratings.max())
    one_hot_ratings = np.eye(y_dim)[ratings.astype(int) - 1]
    
    # Combine embeddings and ratings
    combined_data = np.concatenate([embeddings, one_hot_ratings], axis=1)
    return combined_data, embeddings.shape[1], y_dim


def main():
    """Main training workflow"""
    parser = argparse.ArgumentParser(description="Mutual Information Estimation with MINE")
    parser.add_argument('--emb_path', type=str, required=True, help='Path to embedding numpy file')
    parser.add_argument('--rating_dir', type=str, required=True, help='Directory containing rating CSV files')
    parser.add_argument('--output_prefix', type=str, required=True, help='Output file prefix')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Model save directory')
    parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--hidden_size', type=int, default=100, help='MINE network hidden size')
    parser.add_argument('--log_freq', type=int, default=2000, help='Logging frequency')
    parser.add_argument('--ma_rate', type=float, default=0.01, help='Moving average rate')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    data, x_dim, y_dim = load_data(args.emb_path, args.rating_dir)
    print(f"Data shape: {data.shape}, x_dim: {x_dim}, y_dim: {y_dim}")
    
    # Initialize model
    model = MutualInformationEstimator(x_dim + y_dim, args.hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training configuration
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'x_dim': x_dim,
        'log_freq': args.log_freq,
        'ma_rate': args.ma_rate,
        'save_path': f"{args.save_dir}/{args.output_prefix}_model.pth",
        'output_prefix': args.output_prefix,
        'verbose': True
    }
    
    # Train model
    mi_estimates = train_model(data, model, optimizer, device, config)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(mi_estimates, alpha=0.6, label='Raw Estimates')
    
    window_size = min(1000, len(mi_estimates)//10)
    smoothed = moving_average(mi_estimates, window_size)
    plt.plot(range(window_size, len(mi_estimates)), smoothed, 'r-', linewidth=2, label='Smoothed')
    
    plt.title("Mutual Information Estimation")
    plt.xlabel("Iterations")
    plt.ylabel("MI Estimate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{args.output_prefix}_mi_trend.png")
    print(f"Final MI estimate: {smoothed[-1]:.4f}")


if __name__ == "__main__":
    main()