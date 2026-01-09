import torch
import torch.nn as nn
from typing import Dict


class TokenProjector(nn.Module):
    """
    Project different token types to unified d_model dimension
    with optional learnable token importance weights
    """
    
    def __init__(self, 
                 input_dims: Dict[str, int], 
                 d_model: int, 
                 use_batch_norm: bool = True,
                 use_token_weights: bool = True,
                 init_strategy: str = 'balanced'):
        """
        Initialize TokenProjector with learnable weights
        
        Args:
            input_dims: Dictionary mapping token names to their input dimensions
            d_model: Target dimension for all tokens
            use_batch_norm: Whether to use batch normalization
            use_token_weights: Whether to use learnable token importance weights
            init_strategy: Initialization strategy ('balanced', 'chem_focused', or 'v2')
        """
        super().__init__()
        self.d_model = d_model
        self.token_names = list(input_dims.keys())
        self.use_batch_norm = use_batch_norm
        self.use_token_weights = use_token_weights
        self.init_strategy = init_strategy
        
        # Build projection layers for each token
        self.projectors = nn.ModuleDict()
        for token_name, input_dim in input_dims.items():
            self.projectors[token_name] = self._build_projector(input_dim, d_model, use_batch_norm)
        
        # Learnable token importance weights with different initialization strategies
        if use_token_weights:
            if init_strategy == 'balanced':
                # All tokens start with similar importance
                initial_weights = {
                    'chem': 1.0,           # MPNN embeddings
                    'Morgan': 0.8,         # Morgan fingerprint - increased from 0.3
                    'MACCS': 0.8,          # MACCS keys - increased from 0.3
                    'RDKit_feats': 0.9,    # RDKit descriptors - increased from 0.5
                    'comp': 1.0,           # Composition
                    'phys': 1.0,           # Physical properties
                    'help': 1.0,           # Helper lipid
                    'exp': 1.0             # Experimental conditions
                }
            elif init_strategy == 'chem_focused':
                # Prioritize chemical features, downweight fingerprints
                initial_weights = {
                    'chem': 1.2,           # MPNN embeddings - highest priority
                    'Morgan': 0.4,         # Morgan fingerprint - lower
                    'MACCS': 0.4,          # MACCS keys - lower
                    'RDKit_feats': 0.6,    # RDKit descriptors
                    'comp': 0.8,           # Composition
                    'phys': 0.8,           # Physical properties
                    'help': 0.8,           # Helper lipid
                    'exp': 0.8             # Experimental conditions
                }
            else:  # 'v2' or default strategy
                # V2 strategy: low weights for Morgan/MACCS
                initial_weights = {
                    'chem': 1.0,
                    'Morgan': 0.3,
                    'MACCS': 0.3,
                    'RDKit_feats': 0.5,
                    'comp': 0.8,
                    'phys': 0.8,
                    'help': 0.8,
                    'exp': 0.8
                }
            
            # Convert to tensor
            weight_values = torch.tensor([
                initial_weights.get(name, 0.5) for name in self.token_names
            ])
            
            # Make weights learnable (will be optimized during training)
            self.token_weights = nn.Parameter(weight_values)
            
            print(f"\n[TokenProjector] Initialized with '{init_strategy}' strategy:")
            for name, weight in zip(self.token_names, weight_values):
                print(f"  {name:15s}: {weight:.3f}")
            print()
    
    def _build_projector(self, input_dim: int, output_dim: int, use_batch_norm: bool) -> nn.Module:
        """
        Build projection network for a single token type
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (d_model)
            use_batch_norm: Whether to use batch normalization
        
        Returns:
            Sequential projection network
        """
        layers = []
        
        # Input normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(input_dim))
        
        # Main projection
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.ReLU())
        
        # Output normalization and regularization
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_dim))
        
        layers.append(nn.Dropout(0.1))
        
        return nn.Sequential(*layers)
    
    def forward(self, token_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Project and weight tokens
        
        Args:
            token_embeddings: Dictionary of {token_name: tensor[batch, feature_dim]}
        
        Returns:
            tensor[batch, num_tokens, d_model]
        """
        projected = []
        
        for i, token_name in enumerate(self.token_names):
            if token_name not in token_embeddings:
                continue
            
            # Get token embedding
            embedding = token_embeddings[token_name]  # [batch, input_dim]
            
            # Project to d_model dimension
            proj = self.projectors[token_name](embedding)  # [batch, d_model]
            
            # Apply learnable importance weight
            if self.use_token_weights:
                # Use sigmoid to bound weights in [0, 1]
                weight = torch.sigmoid(self.token_weights[i])
                proj = proj * weight
            
            # Check for NaN (debugging)
            if torch.isnan(proj).any():
                print(f"[Warning] NaN detected in projected token: {token_name}")
            
            # Add token dimension: [batch, d_model] -> [batch, 1, d_model]
            proj = proj.unsqueeze(1)
            projected.append(proj)
        
        # Concatenate all tokens along token dimension
        # Result: [batch, num_tokens, d_model]
        return torch.cat(projected, dim=1)
    
    def get_token_weights(self):
        """
        Get current token importance weights (for analysis and monitoring)
        
        Returns:
            Dictionary mapping token names to their current weights
        """
        if self.use_token_weights:
            # Apply sigmoid and convert to dict
            weights = torch.sigmoid(self.token_weights).detach().cpu()
            return {name: w.item() for name, w in zip(self.token_names, weights)}
        return None
    
    def print_token_weights(self):
        """Print current token weights with enhanced visualization"""
        weights = self.get_token_weights()
        if weights:
            print("\n" + "="*60)
            print("Token Importance Weights (after sigmoid)")
            print("="*60)
            
            # Sort by weight for easier reading
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            
            for token_name, weight in sorted_weights:
                # Visual bar (scaled to 40 chars)
                bar_length = int(weight * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                print(f"  {token_name:20s} {weight:.4f} |{bar}|")
            
            print("="*60)
            
            # Statistics
            import numpy as np
            weights_list = list(weights.values())
            print(f"  Mean: {np.mean(weights_list):.4f}, "
                  f"Std: {np.std(weights_list):.4f}, "
                  f"Min: {min(weights_list):.4f}, "
                  f"Max: {max(weights_list):.4f}")
            print("="*60 + "\n")