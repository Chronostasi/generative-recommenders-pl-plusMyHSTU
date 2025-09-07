"""
Simple configuration for ranking model
Replaces complex Hydra configuration with Python dictionaries
"""
import torch

# Default configuration for ranking model
RANKING_CONFIG = {
    # Model parameters
    "item_embedding_dim": 50,
    "max_sequence_length": 50,
    "gr_output_length": 10,
    "num_ratings": 6,
    "dropout_rate": 0.2,
    
    # HSTU encoder parameters
    "num_blocks": 2,
    "num_heads": 1,
    "attention_dim": 50,
    "linear_dim": 50,
    "linear_dropout_rate": 0.2,
    "attn_dropout_rate": 0.0,
    
    # Training parameters
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "max_epochs": 500,
    "patience": 20,
    
    # Loss parameters
    "temperature": 0.05,
    
    # Data parameters
    "data_path": "data/ml-1m",
    "max_item_id": 3953,  # Will be updated from data
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

def get_config():
    """Get the default configuration"""
    import torch
    config = RANKING_CONFIG.copy()
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config

def update_config(config, **kwargs):
    """Update configuration with new values"""
    for key, value in kwargs.items():
        config[key] = value
    return config