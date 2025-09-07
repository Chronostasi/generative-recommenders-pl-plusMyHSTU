"""
Embedding modules for ranking model
"""
import torch


def truncated_normal(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    """Initialize tensor with truncated normal distribution"""
    with torch.no_grad():
        tensor.normal_(mean, std)
        # Truncate to 2 standard deviations
        tensor.clamp_(min=mean - 2 * std, max=mean + 2 * std)


class LocalEmbeddingModule(torch.nn.Module):
    """Local item embedding module"""
    
    def __init__(self, num_items: int, item_embedding_dim: int):
        super().__init__()
        self._item_embedding_dim = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def reset_params(self):
        """Initialize parameters"""
        truncated_normal(self._item_emb.weight.data, mean=0.0, std=0.02)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for item IDs"""
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim