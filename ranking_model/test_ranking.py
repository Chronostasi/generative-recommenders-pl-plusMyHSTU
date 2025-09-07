"""
Simple test script to verify ranking model functionality
"""
import os
import sys
import torch
import pandas as pd
import numpy as np

# Add ranking_model to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.ranking import SimpleRankingModel
from data.dataset import SimpleRecoDataset
from utils.features import seq_features_from_row
from torch.utils.data import DataLoader


def create_toy_data():
    """Create toy dataset for testing"""
    data = []
    
    # Create simple sequences for 3 users
    for user_id in range(1, 4):
        for i in range(5):
            data.append({
                'user_id': user_id,
                'item_id': user_id * 10 + i + 1,  # Different items per user
                'rating': (i % 5) + 1,  # Ratings 1-5
                'timestamp': user_id * 1000 + i + 1,
            })
    
    return pd.DataFrame(data)


def test_model_creation():
    """Test model creation and basic forward pass"""
    print("Testing model creation...")
    
    # Get config
    config = get_config()
    config["max_item_id"] = 50  # Small number for testing
    
    # Create model
    model = SimpleRankingModel(config)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, config


def test_data_loading():
    """Test data loading"""
    print("Testing data loading...")
    
    # Create toy data
    toy_df = create_toy_data()
    print(f"✓ Created toy dataset with {len(toy_df)} interactions")
    
    # Create dataset
    dataset = SimpleRecoDataset(
        toy_df,
        max_sequence_length=10,
        ignore_last_n=0,
    )
    print(f"✓ Dataset created with {len(dataset)} sequences")
    
    # Create data loader
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    print(f"✓ DataLoader created with {len(data_loader)} batches")
    
    return data_loader


def test_forward_pass(model, data_loader, config):
    """Test forward pass"""
    print("Testing forward pass...")
    
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in data_loader:
            # Convert batch to sequential features
            seq_features, target_ids, target_ratings = seq_features_from_row(
                batch,
                device=device,
                max_output_length=config["gr_output_length"] + 1,
            )
            
            # Get item embeddings
            input_embeddings = model.embeddings.get_item_embeddings(seq_features.past_ids)
            seq_features = seq_features._replace(past_embeddings=input_embeddings)
            
            # Forward pass
            logits = model.get_logits(seq_features)
            predictions = model.predict(seq_features)
            loss = model.compute_loss(seq_features, target_ratings)
            
            print(f"✓ Forward pass successful:")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Loss: {loss.item():.4f}")
            
            break  # Test only first batch
    
    print("✓ Forward pass completed successfully")


def main():
    """Run all tests"""
    print("Starting ranking model tests...\n")
    
    try:
        # Test model creation
        model, config = test_model_creation()
        print()
        
        # Test data loading
        data_loader = test_data_loading()
        print()
        
        # Test forward pass
        test_forward_pass(model, data_loader, config)
        print()
        
        print("🎉 All tests passed! Ranking model is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()