"""
Demo script to show the complete ranking model workflow
Can be run directly: python demo.py
"""
import os
import sys
import torch
import pandas as pd
import numpy as np

# Add ranking_model to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, update_config
from models.ranking import SimpleRankingModel
from data.dataset import SimpleRecoDataset
from utils.features import seq_features_from_row
from torch.utils.data import DataLoader


def create_sample_data():
    """Create sample MovieLens-like data for demo"""
    print("Creating sample data...")
    
    np.random.seed(42)
    data = []
    
    # Create data for 20 users with different preferences
    for user_id in range(1, 21):
        # Each user has 10-20 interactions
        n_interactions = np.random.randint(10, 21)
        
        # Users have different item preferences
        if user_id <= 5:
            # Users 1-5 like action movies (items 1-100)
            item_pool = list(range(1, 101))
            base_rating = 4  # Higher ratings for preferred genre
        elif user_id <= 10:
            # Users 6-10 like comedy movies (items 101-200)
            item_pool = list(range(101, 201))
            base_rating = 4
        elif user_id <= 15:
            # Users 11-15 like drama movies (items 201-300)
            item_pool = list(range(201, 301))
            base_rating = 3
        else:
            # Users 16-20 have diverse tastes (items 1-300)
            item_pool = list(range(1, 301))
            base_rating = 3
        
        # Generate interactions for this user
        user_items = np.random.choice(item_pool, n_interactions, replace=False)
        
        for i, item_id in enumerate(user_items):
            # Rating varies around base rating (1-5 scale)
            rating = max(1, min(5, base_rating + np.random.randint(-1, 2)))
            # Convert to 0-indexed for embedding layer
            rating_idx = rating - 1  # Convert 1-5 to 0-4
            timestamp = 1000000 + user_id * 1000 + i
            
            data.append({
                'user_id': user_id,
                'item_id': item_id,
                'rating': rating_idx,  # Use 0-indexed rating
                'timestamp': timestamp,
            })
    
    df = pd.DataFrame(data)
    print(f"✓ Created sample dataset with {len(df)} interactions")
    print(f"  Users: {df['user_id'].nunique()}")
    print(f"  Items: {df['item_id'].nunique()}")
    print(f"  Rating distribution: {dict(df['rating'].value_counts().sort_index())}")
    
    return df


def demo_training(sample_df, output_dir="demo_outputs"):
    """Demo training process"""
    print("\n" + "="*50)
    print("DEMO: Training Process")
    print("="*50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get configuration
    config = get_config()
    config = update_config(config, 
                          max_epochs=5,  # Short for demo
                          batch_size=8,
                          patience=3,
                          max_item_id=sample_df['item_id'].max())
    
    device = torch.device("cpu")  # Use CPU for demo
    print(f"Using device: {device}")
    
    # Split data into train/val
    users = sample_df['user_id'].unique()
    np.random.shuffle(users)
    
    train_users = users[:12]  # 12 users for training
    val_users = users[12:]    # 8 users for validation
    
    train_df = sample_df[sample_df['user_id'].isin(train_users)]
    val_df = sample_df[sample_df['user_id'].isin(val_users)]
    
    print(f"Train data: {len(train_df)} interactions from {len(train_users)} users")
    print(f"Val data: {len(val_df)} interactions from {len(val_users)} users")
    
    # Create datasets
    train_dataset = SimpleRecoDataset(train_df, max_sequence_length=config["max_sequence_length"])
    val_dataset = SimpleRecoDataset(val_df, max_sequence_length=config["max_sequence_length"])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    
    print(f"Train sequences: {len(train_dataset)}")
    print(f"Val sequences: {len(val_dataset)}")
    
    # Create model
    model = SimpleRankingModel(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    print("\nTraining...")
    for epoch in range(config["max_epochs"]):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Convert batch to sequential features
            seq_features, target_ids, target_ratings = seq_features_from_row(
                batch, device=device, max_output_length=config["gr_output_length"] + 1
            )
            
            # Get item embeddings
            input_embeddings = model.embeddings.get_item_embeddings(seq_features.past_ids)
            seq_features = seq_features._replace(past_embeddings=input_embeddings)
            
            # Forward pass
            loss = model.compute_loss(seq_features, target_ratings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
    
    # Save model
    checkpoint_path = os.path.join(output_dir, "demo_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
    }, checkpoint_path)
    
    print(f"✓ Model saved to {checkpoint_path}")
    return checkpoint_path, val_loader, config


def demo_evaluation(checkpoint_path, val_loader, config):
    """Demo evaluation process"""
    print("\n" + "="*50)
    print("DEMO: Evaluation Process")
    print("="*50)
    
    device = torch.device("cpu")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleRankingModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    
    # Evaluate
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            # Convert batch to sequential features
            seq_features, target_ids, target_ratings = seq_features_from_row(
                batch, device=device, max_output_length=config["gr_output_length"] + 1
            )
            
            # Get item embeddings
            input_embeddings = model.embeddings.get_item_embeddings(seq_features.past_ids)
            seq_features = seq_features._replace(past_embeddings=input_embeddings)
            
            # Predict
            predictions = model.predict(seq_features)
            predicted_ratings = torch.argmax(predictions, dim=1)
            
            # Calculate accuracy
            target_ratings_flat = target_ratings.squeeze(-1)
            correct = (predicted_ratings == target_ratings_flat).sum().item()
            total_correct += correct
            total_samples += len(target_ratings_flat)
            
            all_predictions.extend(predicted_ratings.cpu().numpy())
            all_targets.extend(target_ratings_flat.cpu().numpy())
    
    accuracy = total_correct / total_samples
    print(f"Validation accuracy: {accuracy:.3f}")
    
    # Show rating distribution
    pred_dist = pd.Series(all_predictions).value_counts().sort_index()
    true_dist = pd.Series(all_targets).value_counts().sort_index()
    
    print("\nRating distribution comparison (0-indexed):")
    print(f"{'Rating':<8} {'True':<8} {'Predicted':<10}")
    print("-" * 25)
    for rating in range(5):  # 0-4 ratings
        true_count = true_dist.get(rating, 0)
        pred_count = pred_dist.get(rating, 0)
        print(f"{rating}({rating+1})<-  {true_count:<8} {pred_count:<10}")  # Show both 0-indexed and 1-5 scale


def demo_prediction(checkpoint_path, config):
    """Demo prediction process"""
    print("\n" + "="*50)
    print("DEMO: Prediction Process")
    print("="*50)
    
    device = torch.device("cpu")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleRankingModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create sample prediction data (using 0-indexed ratings)
    sample_data = [
        [100, 50, 3, 2000001],   # User 100, liked item 50 (rating 4 -> 3)
        [100, 75, 4, 2000002],   # Really liked item 75 (rating 5 -> 4)
        [100, 120, 2, 2000003],  # Moderate rating for item 120 (rating 3 -> 2)
        [100, 150, 3, 2000004],  # Target: what will user rate item 150? (rating 4 -> 3)
    ]
    
    pred_df = pd.DataFrame(sample_data, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    print("Sample user sequence:")
    print(pred_df)
    
    # Create dataset
    pred_dataset = SimpleRecoDataset(pred_df, max_sequence_length=config["max_sequence_length"])
    pred_loader = DataLoader(pred_dataset, batch_size=1, shuffle=False)
    
    # Make prediction
    with torch.no_grad():
        for batch in pred_loader:
            seq_features, target_ids, target_ratings = seq_features_from_row(
                batch, device=device, max_output_length=config["gr_output_length"] + 1
            )
            
            input_embeddings = model.embeddings.get_item_embeddings(seq_features.past_ids)
            seq_features = seq_features._replace(past_embeddings=input_embeddings)
            
            predictions = model.predict(seq_features)
            predicted_rating = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0, predicted_rating].item()
            
            print(f"\nPrediction for item {target_ids.item()}:")
            print(f"Predicted rating: {predicted_rating + 1} (model output: {predicted_rating})")  # Convert back to 1-5 scale
            print(f"True rating: {target_ratings.item() + 1} (model input: {target_ratings.item()})")  # Convert back to 1-5 scale
            print(f"Confidence: {confidence:.3f}")
            print(f"Correct: {predicted_rating == target_ratings.item()}")
            
            # Show all rating probabilities
            print("\nAll rating probabilities:")
            for i in range(len(predictions[0])):
                print(f"  Rating {i+1}: {predictions[0, i].item():.3f}")  # Convert to 1-5 scale for display


def main():
    """Run complete demo"""
    print("🎬 Ranking Model Demo")
    print("This demo shows the complete workflow:")
    print("1. Create sample data")
    print("2. Train model")
    print("3. Evaluate model")
    print("4. Make predictions")
    print("-" * 50)
    
    # Create sample data
    sample_df = create_sample_data()
    
    # Demo training
    checkpoint_path, val_loader, config = demo_training(sample_df)
    
    # Demo evaluation
    demo_evaluation(checkpoint_path, val_loader, config)
    
    # Demo prediction
    demo_prediction(checkpoint_path, config)
    
    print("\n" + "="*50)
    print("🎉 Demo completed successfully!")
    print("="*50)
    print("\nThe ranking model has been successfully extracted and can run directly!")
    print("Key achievements:")
    print("✓ Standalone execution (no make required)")
    print("✓ Simple Python configuration (no Hydra)")
    print("✓ Direct script execution")
    print("✓ Modular architecture")
    print("✓ Complete training/evaluation/prediction workflow")
    print("\nTry running the individual scripts:")
    print("  python train_ranking.py --help")
    print("  python eval_ranking.py --help")
    print("  python predict_ranking.py --help")


if __name__ == "__main__":
    main()