"""
Preprocessor for ranking model - combines item and rating features
"""
import math
import torch


def truncated_normal(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    """Initialize tensor with truncated normal distribution"""
    with torch.no_grad():
        tensor.normal_(mean, std)
        # Truncate to 2 standard deviations
        tensor.clamp_(min=mean - 2 * std, max=mean + 2 * std)


class SequentialFeatures:
    """Simplified sequential features"""
    def __init__(self, past_lengths, past_ids, past_embeddings, past_payloads):
        self.past_lengths = past_lengths
        self.past_ids = past_ids
        self.past_embeddings = past_embeddings
        self.past_payloads = past_payloads
    
    def _replace(self, **kwargs):
        """Create new instance with replaced fields"""
        new_vals = {
            'past_lengths': self.past_lengths,
            'past_ids': self.past_ids, 
            'past_embeddings': self.past_embeddings,
            'past_payloads': self.past_payloads,
        }
        new_vals.update(kwargs)
        return SequentialFeatures(**new_vals)


class CombinedItemAndRatingPreprocessor(torch.nn.Module):
    """Combined item and rating input features preprocessor"""
    
    def __init__(
        self,
        max_sequence_len: int,
        embedding_dim: int,
        dropout_rate: float,
        num_ratings: int,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._pos_emb = torch.nn.Embedding(
            max_sequence_len * 2,
            self._embedding_dim,
        )
        self._dropout_rate = dropout_rate
        self._emb_dropout = torch.nn.Dropout(p=dropout_rate)
        self._rating_emb = torch.nn.Embedding(
            num_ratings,
            self._embedding_dim,
        )
        self.reset_state()

    @property
    def ratings_emb(self) -> torch.Tensor:
        """Get rating embeddings weight matrix"""
        return self._rating_emb.weight

    def reset_state(self) -> None:
        """Initialize parameters"""
        truncated_normal(
            self._pos_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )
        truncated_normal(
            self._rating_emb.weight.data,
            mean=0.0,
            std=math.sqrt(1.0 / self._embedding_dim),
        )

    def forward(self, seq_features: SequentialFeatures) -> torch.Tensor:
        """Process sequential features"""
        # Get item embeddings
        item_embeddings = seq_features.past_embeddings  # [B, N, D]
        
        # Get rating embeddings  
        ratings = seq_features.past_payloads["ratings"]  # [B, N]
        rating_embeddings = self._rating_emb(ratings)  # [B, N, D]
        
        # Get positional embeddings
        B, N = item_embeddings.shape[:2]
        positions = torch.arange(N, device=item_embeddings.device).unsqueeze(0).expand(B, -1)
        pos_embeddings = self._pos_emb(positions)  # [B, N, D]
        
        # Combine all embeddings
        combined_embeddings = item_embeddings + rating_embeddings + pos_embeddings
        
        # Apply dropout
        combined_embeddings = self._emb_dropout(combined_embeddings)
        
        return combined_embeddings