"""
Standalone prediction script for ranking model
Can be run directly: python predict_ranking.py --checkpoint best_model.pt --input sample_input.csv
"""
import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add ranking_model to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from models.ranking import SimpleRankingModel
from data.dataset import SimpleRecoDataset
from utils.features import seq_features_from_row
from torch.utils.data import DataLoader


def create_prediction_dataset(input_data, config):
    """Create dataset from input data for prediction"""
    
    if isinstance(input_data, str):
        # Load from file
        if input_data.endswith('.csv'):
            df = pd.read_csv(input_data)
        else:
            raise ValueError("Input file must be CSV format")
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("Input must be a CSV file path or DataFrame")
    
    # Create dataset
    dataset = SimpleRecoDataset(
        df,
        max_sequence_length=config["max_sequence_length"],
        ignore_last_n=0,
    )
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )
    
    return data_loader, df


def predict_ratings(model, data_loader, device, config):
    """Generate rating predictions"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_sequence_info = []
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Predicting")
        for batch_idx, batch in enumerate(pbar):
            # Convert batch to sequential features
            seq_features, target_ids, target_ratings = seq_features_from_row(
                batch,
                device=device,
                max_output_length=config["gr_output_length"] + 1,
            )
            
            # Get item embeddings
            input_embeddings = model.embeddings.get_item_embeddings(seq_features.past_ids)
            seq_features = seq_features._replace(past_embeddings=input_embeddings)
            
            # Get predictions
            probs = model.predict(seq_features)  # [B, num_ratings]
            predicted_ratings = torch.argmax(probs, dim=1)  # [B]
            
            # Store results
            all_predictions.append(predicted_ratings.cpu())
            all_probabilities.append(probs.cpu())
            
            # Store sequence info for output
            batch_info = {
                'target_item': target_ids.squeeze(-1).cpu(),
                'true_rating': target_ratings.squeeze(-1).cpu(),
                'history_length': seq_features.past_lengths.cpu(),
                'last_item': torch.gather(
                    seq_features.past_ids, 
                    1, 
                    (seq_features.past_lengths - 1).unsqueeze(1)
                ).squeeze(1).cpu(),
            }
            all_sequence_info.append(batch_info)
    
    # Concatenate all results
    predictions = torch.cat(all_predictions, dim=0)
    probabilities = torch.cat(all_probabilities, dim=0)
    
    # Concatenate sequence info
    sequence_info = {}
    for key in all_sequence_info[0].keys():
        sequence_info[key] = torch.cat([info[key] for info in all_sequence_info], dim=0)
    
    return predictions, probabilities, sequence_info


def create_sample_input(output_file="sample_input.csv"):
    """Create a sample input file for demonstration"""
    
    # Sample data with user interaction sequences
    sample_data = [
        # User 1: likes action movies (high ratings)
        [1, 1, 5, 100001],  # User 1, Item 1, Rating 5, Timestamp
        [1, 2, 4, 100002],
        [1, 3, 5, 100003],
        [1, 4, 3, 100004],  # Last interaction to predict
        
        # User 2: likes comedies (medium ratings)
        [2, 10, 3, 200001],
        [2, 11, 4, 200002], 
        [2, 12, 3, 200003],
        [2, 13, 4, 200004],  # Last interaction to predict
        
        # User 3: diverse preferences
        [3, 20, 2, 300001],
        [3, 21, 5, 300002],
        [3, 22, 3, 300003],
        [3, 23, 4, 300004],  # Last interaction to predict
    ]
    
    df = pd.DataFrame(sample_data, columns=['user_id', 'item_id', 'rating', 'timestamp'])
    df.to_csv(output_file, index=False)
    print(f"Sample input file created: {output_file}")
    return output_file


def format_output(predictions, probabilities, sequence_info, rating_offset=1):
    """Format predictions into a readable DataFrame"""
    
    results = []
    
    for i in range(len(predictions)):
        # Adjust rating back (remove offset)
        pred_rating = predictions[i].item() - rating_offset + 1
        true_rating = sequence_info['true_rating'][i].item() - rating_offset + 1
        
        # Get top-3 rating probabilities
        probs = probabilities[i].numpy()
        top_ratings = np.argsort(probs)[-3:][::-1]  # Top 3 in descending order
        
        result = {
            'target_item': sequence_info['target_item'][i].item(),
            'predicted_rating': pred_rating,
            'true_rating': true_rating,
            'correct': pred_rating == true_rating,
            'history_length': sequence_info['history_length'][i].item(),
            'last_item': sequence_info['last_item'][i].item(),
            'confidence': probs[predictions[i]],
        }
        
        # Add top-3 predictions with probabilities
        for j, rating_idx in enumerate(top_ratings):
            actual_rating = rating_idx - rating_offset + 1
            result[f'top_{j+1}_rating'] = actual_rating
            result[f'top_{j+1}_prob'] = probs[rating_idx]
        
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Predict with ranking model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--input", type=str, default=None,
                       help="Input CSV file with user sequences")
    parser.add_argument("--output", type=str, default="predictions.csv",
                       help="Output CSV file for predictions")
    parser.add_argument("--create_sample", action="store_true",
                       help="Create sample input file")
    
    args = parser.parse_args()
    
    # Create sample input if requested
    if args.create_sample:
        sample_file = create_sample_input("sample_input.csv")
        if args.input is None:
            args.input = sample_file
            print(f"Using created sample file: {sample_file}")
    
    if args.input is None:
        print("Error: Please provide input file with --input or use --create_sample")
        return
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Setup device
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Create model and load weights
    print("Creating model...")
    model = SimpleRankingModel(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load input data
    print(f"Loading input data from {args.input}...")
    data_loader, input_df = create_prediction_dataset(args.input, config)
    print(f"Loaded {len(data_loader.dataset)} sequences for prediction")
    
    # Generate predictions
    print("Generating predictions...")
    predictions, probabilities, sequence_info = predict_ratings(
        model, data_loader, device, config
    )
    
    # Format output
    results_df = format_output(predictions, probabilities, sequence_info)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total predictions: {len(results_df)}")
    print(f"Accuracy: {results_df['correct'].mean():.3f}")
    print(f"Average confidence: {results_df['confidence'].mean():.3f}")
    
    # Show rating distribution
    print("\nPredicted rating distribution:")
    rating_dist = results_df['predicted_rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"  Rating {rating}: {count} predictions")
    
    # Show first few predictions
    print(f"\nFirst 5 predictions:")
    print(results_df[['target_item', 'predicted_rating', 'true_rating', 'correct', 'confidence']].head())
    
    print(f"\nPrediction completed! Results saved to {args.output}")


if __name__ == "__main__":
    main()