import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class LNPMixAttentionModel(nn.Module):
    # Complete LNPMix model with cross-modal attention
    # 
    # Architecture:
    #   1. Encode 8 tokens from various sources
    #   2. Project all tokens to d_model dimension
    #   3. Split into Channel-A [Morgan, MACCS, RDKit_feats, chem] and Channel-B [comp, phys, help, exp]
    #   4. Apply cross-modal attention between channels
    #   5. Fuse attended tokens
    #   6. Multi-task prediction heads
    #
    # Input shapes per token:
    #   chem: [B, 300] (MPNN embedding)
    #   Morgan: [B, 1024]
    #   MACCS: [B, 167]
    #   RDKit_feats: [B, 210]
    #   comp: [B, n_comp_features]
    #   phys: [B, n_phys_features]
    #   help: [B, n_help_features]
    #   exp: [B, n_exp_features]
    #
    # Output: Dict[task_name, Tensor[B, 1]]
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_attn_layers = config.num_attention_layers
        
        # Import encoder modules
        from models.encoders.mpnn_encoder import MPNNEncoder
        from models.encoders.rdkit_encoder import RDKitEncoder
        from models.encoders.tabular_encoder import TabularEncoder
        from models.layers.token_projector import TokenProjector
        from models.layers.cross_modal_attention import BiDirectionalCrossAttention
        
        # 1. Encoders
        self.mpnn_encoder = MPNNEncoder(
            checkpoint_path=config.mpnn_checkpoint_path,
            freeze=config.mpnn_freeze,
            output_dim=config.mpnn_output_dim
        ) if not config.disable_mpnn_token else None
        
        self.rdkit_encoder = RDKitEncoder(
            rdkit_features_path=config.rdkit_features_path,
            morgan_dim=config.morgan_dim,
            maccs_dim=config.maccs_dim,
            rdkit_desc_dim=config.rdkit_feats_dim
        )
        
        # FIX: Use correct parameter names for TabularEncoder
        self.tabular_encoder = TabularEncoder(
            comp_cols=config.data_grouping['comp'],
            phys_cols=config.get_phys_cols_with_pdi(),  # Include processed PDI
            help_prefixes=config.data_groups.get('help_prefixes', []),
            exp_prefixes=config.data_groups.get('exp_prefixes', [])
        )
        
        # 2. Token Projector
        # IMPORTANT: Use actual output dimensions from encoders, not config values
        input_dims = {
            'chem': self.mpnn_encoder.get_output_dim() if self.mpnn_encoder is not None else config.mpnn_output_dim,
            'Morgan': config.morgan_dim,
            'MACCS': config.maccs_dim,
            'RDKit_feats': config.rdkit_feats_dim,
            'comp': len(config.data_grouping['comp']),
            'phys': len(config.get_phys_cols_with_pdi()),
            'help': len(config.data_grouping['help']),
            'exp': len(config.data_grouping['exp'])
        }
        
        print(f"[LNPMixModel] Token input dimensions:")
        for name, dim in input_dims.items():
            print(f"  {name}: {dim}")
        
        self.token_projector = TokenProjector(
            input_dims=input_dims,
            d_model=config.d_model,
            use_batch_norm=True,
            use_token_weights=getattr(config, 'use_token_weights', True),
            init_strategy=getattr(config, 'token_weight_init_strategy', 'balanced')
        )
        
        # 3. Cross-Modal Attention Layers
        self.cross_attn_layers = nn.ModuleList([
            BiDirectionalCrossAttention(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dropout=config.dropout,
                pre_norm=True,
                residual=True
            ) for _ in range(self.num_attn_layers)
        ])
        
        # 4. Fusion Layer
        # Options: concat, mean, max, attention-weighted
        self.fusion_type = config.fusion_type
        if self.fusion_type == 'concat':
            self.fusion_dim = 8 * self.d_model
        elif self.fusion_type in ['mean', 'max']:
            self.fusion_dim = self.d_model
        elif self.fusion_type == 'attention':
            # Learnable attention weights for 8 tokens
            self.fusion_attention = nn.Sequential(
                nn.Linear(self.d_model, 1),
                nn.Softmax(dim=1)
            )
            self.fusion_dim = self.d_model
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # 5. Multi-Task Prediction Heads with separate dropouts
        self.task_names = config.task_names
        self.task_types = config.task_types  # Dict[task_name, 'regression' or 'classification']
        
        # Get task-specific dropouts
        clf_dropout = getattr(config, 'classification_dropout', config.dropout)
        reg_dropout = getattr(config, 'regression_dropout', config.dropout)
        
        # Separate heads by task type
        self.regression_tasks = [t for t, type in self.task_types.items() if type == 'regression']
        self.classification_tasks = [t for t, type in self.task_types.items() if type == 'classification']
        
        print(f"\n[LNPMixModel] Building task heads:")
        print(f"  Regression tasks: {len(self.regression_tasks)}")
        print(f"  Classification tasks: {len(self.classification_tasks)}")
        print(f"  Regression dropout: {reg_dropout:.3f}")
        print(f"  Classification dropout: {clf_dropout:.3f}")
        
        self.heads = nn.ModuleDict()
        for task_name in self.task_names:
            task_type = self.task_types[task_name]
            dropout = clf_dropout if task_type == 'classification' else reg_dropout
            
            # MLP head with task-specific dropout
            layers = []
            prev_dim = self.fusion_dim
            for hidden_dim in config.head_hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)  # Use task-specific dropout
                ])
                prev_dim = hidden_dim
            
            # Output layer
            if task_type == 'regression':
                layers.append(nn.Linear(prev_dim, 1))
            else:  # classification
                layers.append(nn.Linear(prev_dim, 1))
                # Note: Using BCEWithLogitsLoss in get_loss, no sigmoid here
            
            self.heads[task_name] = nn.Sequential(*layers)
        
        print(f"\n[LNPMixAttentionModel] Initialized")
        print(f"  d_model={self.d_model}, num_heads={self.num_heads}")
        print(f"  num_attention_layers={self.num_attn_layers}")
        print(f"  fusion_type={self.fusion_type}, fusion_dim={self.fusion_dim}")
        print(f"  tasks={len(self.task_names)} ({list(self.task_names)})")
    
    def forward(self, smiles: List[str], tabular_data: Dict[str, torch.Tensor], 
                return_attention: bool = False) -> Union[Dict[str, torch.Tensor], 
                                                         Tuple[Dict[str, torch.Tensor], List]]:
        batch_size = len(smiles)
        
        # Step 1: Encode tokens
        token_dict = {}
        
        # MPNN token (chem)
        if self.mpnn_encoder is not None:
            token_dict['chem'] = self.mpnn_encoder(smiles)  # [B, 601]
            if torch.isnan(token_dict['chem']).any():
                print(f"[ERROR] NaN detected in MPNN encoder output!")
                print(f"  chem token shape: {token_dict['chem'].shape}")
        else:
            token_dict['chem'] = torch.zeros(batch_size, self.config.mpnn_output_dim, 
                                            device=next(self.parameters()).device)
        
        # RDKit tokens (Morgan, MACCS, RDKit_feats)
        rdkit_tokens = self.rdkit_encoder(smiles)
        for k, v in rdkit_tokens.items():
            if torch.isnan(v).any():
                print(f"[ERROR] NaN detected in RDKit encoder output: {k}")
        token_dict.update(rdkit_tokens)
        
        # Tabular tokens (comp, phys, help, exp)
        tabular_tokens = self.tabular_encoder(tabular_data)
        for k, v in tabular_tokens.items():
            if torch.isnan(v).any():
                print(f"[ERROR] NaN detected in tabular encoder output: {k}")
        token_dict.update(tabular_tokens)
        
        # Debug: Print detailed info for first batch
        if len(smiles) == 32:  # Assuming first batch has 32 samples
            import os
            if not hasattr(self, '_first_batch_logged'):
                self._first_batch_logged = True
                print("\n[Debug] First batch token statistics:")
                for k, v in token_dict.items():
                    print(f"  {k}: shape={v.shape}, min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}, has_nan={torch.isnan(v).any().item()}")
        
        # Step 2: Project all tokens to d_model
        tokens_projected = self.token_projector(token_dict)  # [B, 8, d_model]
        
        if torch.isnan(tokens_projected).any():
            print(f"[ERROR] NaN detected after token projection!")
            print(f"  tokens_projected shape: {tokens_projected.shape}")
            print(f"  NaN count: {torch.isnan(tokens_projected).sum().item()}")
            # Check which tokens have NaN
            for i, name in enumerate(['chem', 'Morgan', 'MACCS', 'RDKit_feats', 'comp', 'phys', 'help', 'exp']):
                if torch.isnan(tokens_projected[:, i, :]).any():
                    print(f"    Token {i} ({name}) has NaN")
        
        # Step 3: Split into Channel-A and Channel-B
        # Channel-A: [Morgan, MACCS, RDKit_feats, chem] = indices [1, 2, 3, 0]
        # Channel-B: [comp, phys, help, exp] = indices [4, 5, 6, 7]
        chan_a = tokens_projected[:, [1, 2, 3, 0], :]  # [B, 4, d_model]
        chan_b = tokens_projected[:, [4, 5, 6, 7], :]  # [B, 4, d_model]
        
        # Step 4: Apply cross-modal attention
        attention_weights_list = []
        for layer_idx, attn_layer in enumerate(self.cross_attn_layers):
            chan_a, chan_b, attn_shapes = attn_layer(chan_a, chan_b)
            if return_attention:
                attention_weights_list.append(attn_shapes)
        
        # Step 5: Concatenate back to 8 tokens
        # Reorder: [chem, Morgan, MACCS, RDKit_feats, comp, phys, help, exp]
        tokens_attended = torch.cat([
            chan_a[:, [3], :],  # chem
            chan_a[:, [0], :],  # Morgan
            chan_a[:, [1], :],  # MACCS
            chan_a[:, [2], :],  # RDKit_feats
            chan_b[:, [0], :],  # comp
            chan_b[:, [1], :],  # phys
            chan_b[:, [2], :],  # help
            chan_b[:, [3], :]   # exp
        ], dim=1)  # [B, 8, d_model]
        
        # Step 6: Fusion
        if self.fusion_type == 'concat':
            fused = tokens_attended.view(batch_size, -1)  # [B, 8*d_model]
        elif self.fusion_type == 'mean':
            fused = tokens_attended.mean(dim=1)  # [B, d_model]
        elif self.fusion_type == 'max':
            fused = tokens_attended.max(dim=1)[0]  # [B, d_model]
        elif self.fusion_type == 'attention':
            attn_weights = self.fusion_attention(tokens_attended)  # [B, 8, 1]
            fused = (tokens_attended * attn_weights).sum(dim=1)  # [B, d_model]
        
        # Step 7: Multi-task predictions
        predictions = {}
        for task_name in self.task_names:
            predictions[task_name] = self.heads[task_name](fused)  # [B, 1]
        
        if return_attention:
            return predictions, attention_weights_list
        else:
            return predictions
    
    def get_loss(self, 
                 predictions: Dict[str, torch.Tensor],
                 targets: Dict[str, torch.Tensor],
                 mask: Dict[str, torch.Tensor],
                 task_weights: Dict[str, float] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss with optional task-specific weighting
        
        Args:
            predictions: Dict of predictions for each task
            targets: Dict of target values for each task
            mask: Dict of masks indicating valid samples
            task_weights: Optional dict of task-specific loss weights
        
        Returns:
            total_loss: Weighted sum of task losses
            loss_dict: Dictionary of individual task losses
        """
        
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
        loss_dict = {}
        
        # Use config task weights if not provided
        if task_weights is None:
            task_weights = self.config.task_weights if hasattr(self.config, 'task_weights') else {}
        
        # Get label smoothing for classification
        label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
        
        valid_task_count = 0  # Count tasks with valid loss
        
        for task_name in self.task_names:
            pred = predictions[task_name]  # [B, 1]
            target = targets[task_name]  # [B, 1]
            task_mask = mask[task_name]  # [B, 1]
            
            # Skip if no valid samples for this task
            if task_mask.sum() == 0:
                loss_dict[task_name] = 0.0
                continue
            
            # Check for NaN or Inf in predictions or targets
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"[WARNING] NaN or Inf in predictions for {task_name}")
                loss_dict[task_name] = 0.0
                continue
            
            if torch.isnan(target).any() or torch.isinf(target).any():
                print(f"[WARNING] NaN or Inf in targets for {task_name}")
                loss_dict[task_name] = 0.0
                continue
            
            task_type = self.task_types[task_name]
            
            # Compute loss only for valid samples
            if task_type == 'regression':
                loss_fn = nn.MSELoss(reduction='none')
                task_loss = loss_fn(pred, target)
                task_loss = (task_loss * task_mask).sum() / task_mask.sum()
            else:  # classification
                # Use BCEWithLogitsLoss with optional label smoothing
                if label_smoothing > 0:
                    # Apply label smoothing: 0 -> eps, 1 -> 1-eps
                    target_smooth = target * (1 - label_smoothing) + 0.5 * label_smoothing
                    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                    task_loss = loss_fn(pred, target_smooth)
                else:
                    loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                    task_loss = loss_fn(pred, target)
                
                task_loss = (task_loss * task_mask).sum() / task_mask.sum()
            
            # Check if task_loss is NaN
            if torch.isnan(task_loss):
                print(f"[WARNING] NaN loss for {task_name} (type: {task_type})")
                loss_dict[task_name] = 0.0
                continue
            
            # Apply task-specific weight
            weight = task_weights.get(task_name, 1.0)
            weighted_loss = weight * task_loss
            
            total_loss = total_loss + weighted_loss
            loss_dict[task_name] = task_loss.item()
            valid_task_count += 1
        
        # If all tasks have NaN, return a dummy loss with gradient
        if valid_task_count == 0:
            print("[ERROR] All tasks have NaN predictions! Using dummy loss.")
            dummy_loss = sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0 + 1e-6
            return dummy_loss, loss_dict
        
        return total_loss, loss_dict

    def get_token_importance(self):
        return self.token_projector.get_token_weights()

    def print_token_importance(self):
        self.token_projector.print_token_importance()