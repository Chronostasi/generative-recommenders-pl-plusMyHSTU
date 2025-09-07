"""
Standalone evaluation script for ranking model
Can be run directly: python eval_ranking.py --checkpoint best_model.pt
"""
import argparse
import os
import sys
import torch
from tqdm import tqdm
import numpy as np

# Add ranking_model to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.ranking import SimpleRankingModel
from data.dataset import create_data_loaders
from utils.features import seq_features_from_row


def compute_detailed_metrics(predictions, targets, k_list=[1, 5, 10]):
    """Compute detailed ranking metrics"""
    metrics = {}
    
    # Convert to numpy
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Basic accuracy
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == targets)
    metrics['accuracy'] = accuracy
    
    # Top-k accuracy
    for k in k_list:
        if k <= predictions.shape[1]:
            top_k_preds = np.argsort(predictions, axis=1)[:, -k:]
            top_k_acc = np.mean([target in pred for target, pred in zip(targets, top_k_preds)])
            metrics[f'top_{k}_accuracy'] = top_k_acc
    
    # Precision, Recall, F1 for each class
    num_classes = predictions.shape[1]
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for class_idx in range(num_classes):
        # True positives
        tp = np.sum((predicted_classes == class_idx) & (targets == class_idx))
        # False positives  
        fp = np.sum((predicted_classes == class_idx) & (targets != class_idx))
        # False negatives
        fn = np.sum((predicted_classes != class_idx) & (targets == class_idx))
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision_per_class.append(precision)
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_per_class.append(recall)
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1)
    
    metrics['precision_per_class'] = precision_per_class
    metrics['recall_per_class'] = recall_per_class
    metrics['f1_per_class'] = f1_per_class
    metrics['macro_precision'] = np.mean(precision_per_class)
    metrics['macro_recall'] = np.mean(recall_per_class)
    metrics['macro_f1'] = np.mean(f1_per_class)
    
    # Class distribution
    unique_targets, target_counts = np.unique(targets, return_counts=True)
    unique_preds, pred_counts = np.unique(predicted_classes, return_counts=True)
    
    metrics['target_distribution'] = dict(zip(unique_targets, target_counts))
    metrics['prediction_distribution'] = dict(zip(unique_preds, pred_counts))
    
    return metrics


def evaluate_model(model, data_loader, device, config):
    """Evaluate model on data loader"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Evaluating")
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
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_detailed_metrics(all_predictions, all_targets)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics, all_predictions, all_targets


def print_metrics(metrics, title="Metrics"):
    """Print metrics in a nice format"""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Main metrics
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    # Top-k accuracy
    for k in [1, 5, 10]:
        if f'top_{k}_accuracy' in metrics:
            print(f"Top-{k} Accuracy: {metrics[f'top_{k}_accuracy']:.4f}")
    
    # Macro metrics
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    for i, (p, r, f1) in enumerate(zip(
        metrics['precision_per_class'], 
        metrics['recall_per_class'], 
        metrics['f1_per_class']
    )):
        print(f"  Class {i}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}")
    
    # Distribution
    print(f"\nTarget Distribution: {metrics['target_distribution']}")
    print(f"Prediction Distribution: {metrics['prediction_distribution']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranking model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, default="data/ml-1m",
                       help="Path to data directory")
    parser.add_argument("--split", type=str, default="test", 
                       choices=["train", "val", "test"],
                       help="Which split to evaluate on")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Save detailed results to file")
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Setup device
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_path, config
    )
    
    # Select data loader
    if args.split == "train":
        data_loader = train_loader
    elif args.split == "val":
        data_loader = val_loader
    else:
        data_loader = test_loader
    
    print(f"Evaluating on {args.split} set: {len(data_loader.dataset)} samples")
    
    # Create model and load weights
    print("Creating model...")
    model = SimpleRankingModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Evaluate
    print("Starting evaluation...")
    metrics, predictions, targets = evaluate_model(model, data_loader, device, config)
    
    # Print results
    print_metrics(metrics, f"{args.split.capitalize()} Set Results")
    
    # Save detailed results if requested
    if args.output_file:
        print(f"\nSaving detailed results to {args.output_file}...")
        
        results = {
            'metrics': metrics,
            'predictions': predictions.cpu().numpy(),
            'targets': targets.cpu().numpy(),
            'config': config,
            'checkpoint_path': args.checkpoint,
        }
        
        # Save as numpy file for easy loading
        np.savez(args.output_file, **results)
        print(f"Results saved to {args.output_file}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()