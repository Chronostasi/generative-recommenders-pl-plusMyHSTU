# Standalone Ranking Model

This is a simplified, standalone version of the ranking model extracted from the full generative recommendation system. It can be run directly with Python scripts without requiring the complex make-based workflow.

## Overview

The ranking model predicts user ratings for items based on their interaction history. It uses:
- **Item embeddings**: Local embedding module for item representations
- **Sequential encoding**: Simplified HSTU (Hierarchical Sequential Transduction Unit) encoder
- **Rating prediction**: Cross-entropy loss for rating classification

## Directory Structure

```
ranking_model/
├── __init__.py
├── config.py              # Simple configuration system
├── train_ranking.py       # Training script
├── eval_ranking.py        # Evaluation script  
├── predict_ranking.py     # Prediction script
├── README.md              # This file
├── models/                # Model components
│   ├── __init__.py
│   ├── embeddings.py      # Item embedding module
│   ├── preprocessor.py    # Combined item+rating preprocessor
│   ├── hstu.py           # Simplified HSTU encoder
│   └── ranking.py        # Main ranking model
├── data/                  # Data utilities
│   ├── __init__.py
│   └── dataset.py        # Simple dataset and data loading
└── utils/                 # Utilities
    ├── __init__.py
    ├── features.py       # Feature processing utilities
    └── ops.py            # Operation utilities
```

## Quick Start

### 1. Training

Train the ranking model on MovieLens-1M data:

```bash
cd ranking_model
python train_ranking.py --data_path ../data/ml-1m --output_dir outputs
```

Options:
- `--data_path`: Path to MovieLens data directory
- `--output_dir`: Directory to save model checkpoints
- `--max_epochs`: Maximum training epochs (default: 100)
- `--batch_size`: Batch size (default: 128) 
- `--learning_rate`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 10)

### 2. Evaluation

Evaluate a trained model:

```bash
python eval_ranking.py --checkpoint outputs/best_model.pt --data_path ../data/ml-1m
```

Options:
- `--checkpoint`: Path to model checkpoint
- `--data_path`: Path to data directory
- `--split`: Which split to evaluate (train/val/test)
- `--output_file`: Save detailed results to file

### 3. Prediction

Make predictions on new data:

```bash
# Create sample input and make predictions
python predict_ranking.py --checkpoint outputs/best_model.pt --create_sample --output predictions.csv

# Use your own input file
python predict_ranking.py --checkpoint outputs/best_model.pt --input my_data.csv --output results.csv
```

Options:
- `--checkpoint`: Path to model checkpoint
- `--input`: Input CSV file with user sequences
- `--output`: Output CSV file for predictions
- `--create_sample`: Create a sample input file for demo

## Configuration

The model configuration is defined in `config.py`. Key parameters:

```python
RANKING_CONFIG = {
    # Model architecture
    "item_embedding_dim": 50,
    "max_sequence_length": 50,
    "num_ratings": 6,
    "num_blocks": 2,           # HSTU layers
    "num_heads": 1,            # Attention heads
    "attention_dim": 50,
    "linear_dim": 50,
    "dropout_rate": 0.2,
    
    # Training
    "batch_size": 128,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "max_epochs": 500,
    "patience": 20,
    
    # Loss
    "temperature": 0.05,
}
```

## Data Format

### Input Data (CSV)
The input data should be a CSV file with columns:
- `user_id`: User identifier
- `item_id`: Item identifier  
- `rating`: Rating (1-5)
- `timestamp`: Interaction timestamp

### Prediction Input
Same format as training data. The model will predict ratings for the target interactions.

### Prediction Output
The prediction script outputs a CSV with:
- `target_item`: Item to predict rating for
- `predicted_rating`: Predicted rating (1-5)
- `true_rating`: Actual rating (if available)
- `correct`: Whether prediction matches true rating
- `confidence`: Model confidence in prediction
- `top_1_rating`, `top_1_prob`: Top prediction and probability
- `top_2_rating`, `top_2_prob`: Second best prediction
- `top_3_rating`, `top_3_prob`: Third best prediction

## Model Architecture

### Components

1. **LocalEmbeddingModule**: Maps item IDs to dense embeddings
2. **CombinedItemAndRatingPreprocessor**: Combines item embeddings with rating embeddings and positional encodings
3. **SimplifiedHSTU**: Hierarchical sequential encoder with multi-head attention
4. **SimpleRankingModel**: Main model that coordinates all components

### Key Simplifications

Compared to the original complex system, this standalone version:
- Removes Hydra configuration dependency
- Simplifies HSTU to core attention mechanism
- Uses direct Python execution instead of Lightning framework
- Focuses only on ranking (not retrieval)
- Streamlined data loading
- Direct metric computation

## Performance

The model achieves similar performance to the original on MovieLens-1M:
- **Accuracy**: Rating prediction accuracy
- **Top-k Accuracy**: Whether true rating is in top-k predictions
- **Cross-entropy Loss**: Classification loss

## Dependencies

Required packages:
- torch
- pandas  
- numpy
- tqdm

Install with:
```bash
pip install torch pandas numpy tqdm
```

## Differences from Original

This standalone version differs from the original system:

### Simplified
- ✅ Direct Python execution (no make required)
- ✅ Simple configuration (no Hydra)
- ✅ Focused on ranking only
- ✅ Minimal dependencies
- ✅ Easy debugging and modification

### Removed
- ❌ Lightning framework overhead
- ❌ Complex Hydra configuration 
- ❌ Retrieval components
- ❌ Advanced metrics and logging
- ❌ Distributed training support

This makes it ideal for:
- Learning and understanding the ranking model
- Quick experimentation and prototyping
- Debugging and development
- Integration into other systems

## Development

To modify the model:

1. **Architecture changes**: Edit files in `models/`
2. **Data processing**: Modify `data/dataset.py`
3. **Training logic**: Update `train_ranking.py`
4. **Configuration**: Adjust `config.py`

The modular structure makes it easy to experiment with different components while maintaining the core functionality.