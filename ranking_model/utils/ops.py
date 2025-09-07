"""
Utility operations for ranking model
"""
import torch


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