"""
Simple data loading for ranking model
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional
import numpy as np


class SimpleRecoDataset(Dataset):
    """Simple recommendation dataset for ranking"""
    
    def __init__(
        self,
        ratings_df: pd.DataFrame,
        max_sequence_length: int,
        ignore_last_n: int = 0,
        rating_offset: int = 0,  # No offset needed - keep ratings 0-indexed for embedding
    ):
        """
        Args:
            ratings_df: DataFrame with columns ['user_id', 'item_id', 'rating', 'timestamp']
            max_sequence_length: Maximum sequence length
            ignore_last_n: Number of last interactions to ignore
            rating_offset: Offset to add to ratings (should be 0 for proper embedding indexing)
        """
        self.max_sequence_length = max_sequence_length
        self.rating_offset = rating_offset
        
        # Sort by user and timestamp
        self.data = ratings_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        
        # Group by user
        user_groups = self.data.groupby('user_id')
        
        self.sequences = []
        for user_id, group in user_groups:
            # Skip users with too few interactions
            if len(group) <= ignore_last_n + 1:
                continue
                
            # Remove last n interactions
            if ignore_last_n > 0:
                group = group.iloc[:-ignore_last_n]
                
            if len(group) < 2:  # Need at least 2 interactions (history + target)
                continue
                
            items = group['item_id'].values
            ratings = group['rating'].values + self.rating_offset  # Add offset
            timestamps = group['timestamp'].values
            
            # Create sequences (sliding window approach)
            for i in range(1, len(items)):
                history_items = items[:i]
                history_ratings = ratings[:i]  
                history_timestamps = timestamps[:i]
                
                target_item = items[i]
                target_rating = ratings[i]
                target_timestamp = timestamps[i]
                
                self.sequences.append({
                    'history_items': history_items,
                    'history_ratings': history_ratings,
                    'history_timestamps': history_timestamps,
                    'target_item': target_item,
                    'target_rating': target_rating,
                    'target_timestamp': target_timestamp,
                })
                
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad or truncate history
        history_length = len(seq['history_items'])
        
        if history_length > self.max_sequence_length:
            # Truncate to most recent items
            start_idx = history_length - self.max_sequence_length
            history_items = seq['history_items'][start_idx:]
            history_ratings = seq['history_ratings'][start_idx:]
            history_timestamps = seq['history_timestamps'][start_idx:]
            history_length = self.max_sequence_length
        else:
            history_items = seq['history_items']
            history_ratings = seq['history_ratings']
            history_timestamps = seq['history_timestamps']
            
        # Pad sequences
        padded_items = np.zeros(self.max_sequence_length, dtype=np.int64)
        padded_ratings = np.zeros(self.max_sequence_length, dtype=np.int64)
        padded_timestamps = np.zeros(self.max_sequence_length, dtype=np.int64)
        
        padded_items[:history_length] = history_items
        padded_ratings[:history_length] = history_ratings
        padded_timestamps[:history_length] = history_timestamps
        
        return {
            'history_lengths': torch.tensor(history_length, dtype=torch.long),
            'historical_ids': torch.tensor(padded_items, dtype=torch.long),
            'historical_ratings': torch.tensor(padded_ratings, dtype=torch.long),
            'historical_timestamps': torch.tensor(padded_timestamps, dtype=torch.long),
            'target_ids': torch.tensor(seq['target_item'], dtype=torch.long),
            'target_ratings': torch.tensor(seq['target_rating'], dtype=torch.long),
            'target_timestamps': torch.tensor(seq['target_timestamp'], dtype=torch.long),
        }


def load_movielens_data(data_path: str) -> pd.DataFrame:
    """Load MovieLens-1M data"""
    try:
        # Try to load preprocessed data
        ratings_df = pd.read_csv(f"{data_path}/ratings.csv")
        return ratings_df
    except FileNotFoundError:
        # If not found, try to load raw data and preprocess
        try:
            ratings_df = pd.read_csv(
                f"{data_path}/ratings.dat", 
                sep='::', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
            # Save preprocessed version
            ratings_df.to_csv(f"{data_path}/ratings.csv", index=False)
            return ratings_df
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find ratings data in {data_path}")


def create_data_loaders(
    data_path: str,
    config: dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test data loaders"""
    
    # Load data
    ratings_df = load_movielens_data(data_path)
    
    # Update max_item_id in config
    config["max_item_id"] = ratings_df['item_id'].max()
    
    # Split by user to ensure no data leakage
    users = ratings_df['user_id'].unique()
    np.random.shuffle(users)
    
    n_users = len(users)
    n_train = int(n_users * train_ratio)
    n_val = int(n_users * val_ratio)
    
    train_users = users[:n_train]
    val_users = users[n_train:n_train + n_val]
    test_users = users[n_train + n_val:]
    
    train_df = ratings_df[ratings_df['user_id'].isin(train_users)]
    val_df = ratings_df[ratings_df['user_id'].isin(val_users)]
    test_df = ratings_df[ratings_df['user_id'].isin(test_users)]
    
    # Create datasets
    train_dataset = SimpleRecoDataset(
        train_df, 
        max_sequence_length=config["max_sequence_length"],
        ignore_last_n=0,
    )
    val_dataset = SimpleRecoDataset(
        val_df,
        max_sequence_length=config["max_sequence_length"], 
        ignore_last_n=0,
    )
    test_dataset = SimpleRecoDataset(
        test_df,
        max_sequence_length=config["max_sequence_length"],
        ignore_last_n=0,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Simplified for direct execution
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    return train_loader, val_loader, test_loader