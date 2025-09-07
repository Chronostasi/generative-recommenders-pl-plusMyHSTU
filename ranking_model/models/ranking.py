"""
Core ranking model - simplified standalone version
"""
import torch
import torch.nn.functional as F
from typing import Tuple

from .embeddings import LocalEmbeddingModule
from .preprocessor import CombinedItemAndRatingPreprocessor, SequentialFeatures
from .hstu import SimplifiedHSTU


def get_current_embeddings(
    past_lengths: torch.Tensor, seq_embeddings: torch.Tensor
) -> torch.Tensor:
    """Get current embeddings from sequence embeddings based on lengths"""
    B = past_lengths.size(0)
    indices = past_lengths.view(-1, 1, 1).expand(B, 1, seq_embeddings.size(-1)) - 1
    current_embeddings = seq_embeddings.gather(1, indices).squeeze(1)
    return current_embeddings


def normalize_embeddings(embeddings: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize embeddings"""
    return torch.nn.functional.normalize(embeddings, p=2, dim=-1, eps=eps)


class SimpleRankingModel(torch.nn.Module):
    """Simplified ranking model for direct execution"""
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.item_embedding_dim = config["item_embedding_dim"]
        self.max_sequence_length = config["max_sequence_length"]
        self.gr_output_length = config["gr_output_length"]
        
        # Core components
        self.embeddings = LocalEmbeddingModule(
            num_items=config["max_item_id"],
            item_embedding_dim=self.item_embedding_dim,
        )
        
        self.preprocessor = CombinedItemAndRatingPreprocessor(
            max_sequence_len=self.max_sequence_length + self.gr_output_length + 1,
            embedding_dim=self.item_embedding_dim,
            dropout_rate=config["dropout_rate"],
            num_ratings=config["num_ratings"],
        )
        
        self.sequence_encoder = SimplifiedHSTU(
            max_sequence_len=self.max_sequence_length,
            max_output_len=self.gr_output_length + 1,
            embedding_dim=self.item_embedding_dim,
            num_blocks=config["num_blocks"],
            num_heads=config["num_heads"],
            attention_dim=config["attention_dim"],
            linear_dim=config["linear_dim"],
            linear_dropout_rate=config["linear_dropout_rate"],
            attn_dropout_rate=config["attn_dropout_rate"],
        )
        
        # Postprocessor (L2 normalization)
        self.eps = 1e-6
        
        # Loss
        self.temperature = config["temperature"]
        
    def get_ratings_embeddings(self) -> torch.Tensor:
        """Get rating embeddings from preprocessor"""
        return self.preprocessor.ratings_emb
        
    def forward(self, seq_features: SequentialFeatures) -> torch.Tensor:
        """Forward pass through the model"""
        # Preprocess features
        processed_embeddings = self.preprocessor(seq_features)  # [B, N, D]
        
        # Apply sequence encoder
        encoded_embeddings = self.sequence_encoder(processed_embeddings)  # [B, N, D]
        
        # Apply L2 normalization (postprocessor)
        normalized_embeddings = normalize_embeddings(encoded_embeddings, self.eps)
        
        return normalized_embeddings
        
    def get_logits(self, seq_features: SequentialFeatures) -> torch.Tensor:
        """Get logits for ranking"""
        # Forward pass
        seq_embeddings = self.forward(seq_features)  # [B, N, D]
        
        # Get current embeddings (last non-padding position)
        current_embeddings = get_current_embeddings(
            seq_features.past_lengths, seq_embeddings
        )  # [B, D]
        
        # Normalize embeddings
        current_embeddings = normalize_embeddings(current_embeddings, self.eps)
        rating_embeddings = normalize_embeddings(self.get_ratings_embeddings(), self.eps)
        
        # Compute similarity (dot product)
        logits = torch.matmul(current_embeddings, rating_embeddings.t())  # [B, num_ratings]
        
        return logits
        
    def compute_loss(self, seq_features: SequentialFeatures, target_ratings: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for rating prediction"""
        logits = self.get_logits(seq_features)  # [B, num_ratings]
        
        # Apply temperature scaling
        logits = logits / self.temperature
        
        # Cross-entropy loss
        target_ratings = target_ratings.squeeze(-1)  # [B]
        loss = F.cross_entropy(logits, target_ratings)
        
        return loss
        
    def predict(self, seq_features: SequentialFeatures) -> torch.Tensor:
        """Predict rating probabilities"""
        logits = self.get_logits(seq_features)
        probs = F.softmax(logits / self.temperature, dim=-1)
        return probs