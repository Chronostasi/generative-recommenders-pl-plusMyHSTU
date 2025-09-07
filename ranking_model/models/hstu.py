"""
Simplified HSTU (Hierarchical Sequential Transduction Unit) encoder
This is a streamlined version for ranking only
"""
import torch
import torch.nn.functional as F
import math


class SimplifiedAttentionLayer(torch.nn.Module):
    """Simplified attention layer for HSTU"""
    
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.head_dim = attention_dim // num_heads
        
        # Attention projections
        self.q_proj = torch.nn.Linear(embedding_dim, attention_dim)
        self.k_proj = torch.nn.Linear(embedding_dim, attention_dim)
        self.v_proj = torch.nn.Linear(embedding_dim, linear_dim)
        self.o_proj = torch.nn.Linear(linear_dim, embedding_dim)
        
        # Layer normalization
        self.layer_norm1 = torch.nn.LayerNorm(embedding_dim)
        self.layer_norm2 = torch.nn.LayerNorm(embedding_dim)
        
        # Feed forward
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, linear_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(linear_dim, embedding_dim),
        )
        
        self.dropout = torch.nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input embeddings
            attn_mask: (N, N) causal attention mask
        Returns:
            (B, N, D) output embeddings
        """
        B, N, D = x.shape
        
        # Self-attention
        residual = x
        x = self.layer_norm1(x)
        
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.linear_dim // self.num_heads).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, self.linear_dim)
        
        output = self.o_proj(attn_output)
        output = residual + self.dropout(output)
        
        # Feed forward
        residual = output
        output = self.layer_norm2(output)
        output = self.ff(output)
        output = residual + output
        
        return output


class SimplifiedHSTU(torch.nn.Module):
    """Simplified HSTU encoder for ranking model"""
    
    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        linear_dropout_rate: float,
        attn_dropout_rate: float,
    ):
        super().__init__()
        
        self.max_sequence_len = max_sequence_len
        self.max_output_len = max_output_len
        self.embedding_dim = embedding_dim
        
        # Stack of attention layers
        self.layers = torch.nn.ModuleList([
            SimplifiedAttentionLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                attention_dim=attention_dim,
                linear_dim=linear_dim,
                dropout_rate=linear_dropout_rate,
            )
            for _ in range(num_blocks)
        ])
        
        # Causal attention mask
        self.register_buffer(
            "attn_mask",
            torch.triu(
                torch.ones(
                    max_sequence_len + max_output_len,
                    max_sequence_len + max_output_len,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input embeddings
        Returns:
            (B, N, D) output embeddings
        """
        B, N, D = x.shape
        
        # Use causal mask for the sequence length
        mask = self.attn_mask[:N, :N]
        
        # Apply each attention layer
        for layer in self.layers:
            x = layer(x, mask)
            
        return x