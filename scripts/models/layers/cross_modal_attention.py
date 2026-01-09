import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    # Cross-attention between two channels
    # Q from Channel-A, K/V from Channel-B
    # Input:
    #   chan_a: [B, 4, d_model]  # [Morgan, MACCS, RDKit_feats, chem]
    #   chan_b: [B, 4, d_model]  # [comp, phys, help, exp]
    # Output:
    #   chan_a_attended: [B, 4, d_model]
    
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 dropout: float = 0.1,
                 pre_norm: bool = True,
                 residual: bool = True):
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)
        self.pre_norm = pre_norm
        self.residual = residual
        
        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        # Normalization and dropout
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        print(f"[CrossModalAttention] d_model={d_model}, num_heads={num_heads}, d_head={self.d_head}")
    
    def forward(self, chan_a: torch.Tensor, chan_b: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Input:
        #   chan_a: [B, 4, d_model]
        #   chan_b: [B, 4, d_model]
        # Output:
        #   attended: [B, 4, d_model]
        #   shapes: dict of intermediate shapes for inspection
        
        B, L_a, D = chan_a.shape
        _, L_b, _ = chan_b.shape
        
        # Pre-normalization
        if self.pre_norm:
            chan_a_norm = self.norm_a(chan_a)
            chan_b_norm = self.norm_b(chan_b)
        else:
            chan_a_norm = chan_a
            chan_b_norm = chan_b
        
        # Project Q, K, V
        Q = self.W_q(chan_a_norm)  # [B, 4, d_model]
        K = self.W_k(chan_b_norm)  # [B, 4, d_model]
        V = self.W_v(chan_b_norm)  # [B, 4, d_model]
        
        # Reshape for multi-head: [B, num_heads, seq_len, d_head]
        Q = Q.view(B, L_a, self.num_heads, self.d_head).transpose(1, 2)  # [B, H, 4, d_head]
        K = K.view(B, L_b, self.num_heads, self.d_head).transpose(1, 2)  # [B, H, 4, d_head]
        V = V.view(B, L_b, self.num_heads, self.d_head).transpose(1, 2)  # [B, H, 4, d_head]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, 4, 4]
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, 4, 4]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, 4, d_head]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_a, D)  # [B, 4, d_model]
        
        # Output projection
        output = self.W_o(attn_output)  # [B, 4, d_model]
        output = self.dropout(output)
        
        # Residual connection
        if self.residual:
            output = output + chan_a
        
        # Post-normalization
        if not self.pre_norm:
            output = self.norm_a(output)
        
        # Collect shapes for debugging
        shapes = {
            'Q': list(Q.shape),
            'K': list(K.shape),
            'V': list(V.shape),
            'scores': list(scores.shape),
            'attn_weights': list(attn_weights.shape),
            'attn_output': list(attn_output.shape),
            'output': list(output.shape)
        }
        
        return output, shapes


class BiDirectionalCrossAttention(nn.Module):
    # Bi-directional cross-attention: both channels attend to each other
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 pre_norm: bool = True, residual: bool = True):
        super().__init__()
        
        # A attends to B
        self.attn_a2b = CrossModalAttention(d_model, num_heads, dropout, pre_norm, residual)
        # B attends to A
        self.attn_b2a = CrossModalAttention(d_model, num_heads, dropout, pre_norm, residual)
        
        print(f"[BiDirectionalCrossAttention] Initialized")
    
    def forward(self, chan_a: torch.Tensor, chan_b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # Returns: (chan_a_attended, chan_b_attended, shapes)
        
        chan_a_out, shapes_a = self.attn_a2b(chan_a, chan_b)
        chan_b_out, shapes_b = self.attn_b2a(chan_b, chan_a)
        
        shapes = {
            'a2b': shapes_a,
            'b2a': shapes_b
        }
        
        return chan_a_out, chan_b_out, shapes