"""
Standalone training script for ranking model
Can be run directly: python train_ranking.py
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Add ranking_model to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config, update_config
from models.ranking import SimpleRankingModel
from data.dataset import create_data_loaders
from utils.features import seq_features_from_row


def compute_metrics(predictions, targets, k_list=[1, 5, 10]):
    """Compute ranking metrics"""
    metrics = {}
    
    # Convert to numpy for easier computation
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Accuracy (exact match)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == targets)
    metrics['accuracy'] = accuracy
    
    # Top-k accuracy
    for k in k_list:
        if k <= predictions.shape[1]:
            top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
            top_k_acc = np.mean([target in pred for target, pred in zip(targets, top_k_preds)])
            metrics[f'top_{k}_accuracy'] = top_k_acc
    
    # Cross-entropy loss
    ce_loss = F.cross_entropy(torch.tensor(predictions), torch.tensor(targets, dtype=torch.long))
    metrics['ce_loss'] = ce_loss.item()
    
    return metrics


def train_epoch(model, train_loader, optimizer, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
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
        loss = model.compute_loss(seq_features, target_ratings)
        
        # Get predictions for metrics
        with torch.no_grad():
            predictions = model.predict(seq_features)
            all_predictions.append(predictions)
            all_targets.append(target_ratings.squeeze(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


def validate_epoch(model, val_loader, device, config):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
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
            loss = model.compute_loss(seq_features, target_ratings)
            predictions = model.predict(seq_features)
            
            total_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(target_ratings.squeeze(-1))
    
    # Compute epoch metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train ranking model")
    parser.add_argument("--data_path", type=str, default="data/ml-1m", 
                       help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory for checkpoints")
    parser.add_argument("--max_epochs", type=int, default=100,
                       help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    config = update_config(config, 
                          data_path=args.data_path,
                          max_epochs=args.max_epochs,
                          batch_size=args.batch_size,
                          learning_rate=args.learning_rate,
                          patience=args.patience)
    
    # Setup device
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_path, config
    )
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples") 
    print(f"Test: {len(test_loader.dataset)} samples")
    
    # Create model
    print("Creating model...")
    model = SimpleRankingModel(config).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(config["max_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['max_epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, config)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device, config)
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print("Saved best model")
        else:
            patience_counter += 1
            
        if patience_counter >= config["patience"]:
            print(f"Early stopping after {epoch + 1} epochs")
            break
    
    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt'))['model_state_dict'])
    test_metrics = validate_epoch(model, test_loader, device, config)
    
    print(f"Test - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}")
    for k in [1, 5, 10]:
        if f'top_{k}_accuracy' in test_metrics:
            print(f"Test - Top-{k} Acc: {test_metrics[f'top_{k}_accuracy']:.4f}")
    
    print(f"\nTraining completed. Best model saved to {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()