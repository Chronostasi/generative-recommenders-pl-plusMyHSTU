"""
Essential utilities for ranking model
"""
from typing import Dict, NamedTuple, Optional, Tuple
import torch


class SequentialFeatures:
    """Features for sequential recommendation"""
    def __init__(self, past_lengths, past_ids, past_embeddings, past_payloads):
        self.past_lengths = past_lengths  # (B,) x int64
        self.past_ids = past_ids  # (B, N,) x int64. 0 denotes padding
        self.past_embeddings = past_embeddings  # (B, N, D) x float
        self.past_payloads = past_payloads  # Implementation-specific payloads
    
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


def seq_features_from_row(
    row,
    device: torch.device,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    """Convert data row to sequential features"""
    historical_lengths = row["history_lengths"].to(device)  # [B]
    historical_ids = row["historical_ids"].to(device)  # [B, N]
    historical_ratings = row["historical_ratings"].to(device)
    historical_timestamps = row["historical_timestamps"].to(device)
    target_ids = row["target_ids"].to(device).unsqueeze(1)  # [B, 1]
    target_ratings = row["target_ratings"].to(device).unsqueeze(1)
    target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)
    
    if max_output_length > 0:
        B = historical_lengths.size(0)
        historical_ids = torch.cat(
            [
                historical_ids,
                torch.zeros(
                    (B, max_output_length), dtype=historical_ids.dtype, device=device
                ),
            ],
            dim=1,
        )
        historical_ratings = torch.cat(
            [
                historical_ratings,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_ratings.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps = torch.cat(
            [
                historical_timestamps,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_timestamps.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )

    seq_features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,  # Will be filled later
        past_payloads={
            "ratings": historical_ratings,
            "timestamps": historical_timestamps,
        },
    )
    return seq_features, target_ids, target_ratings