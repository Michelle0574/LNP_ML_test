import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path


class RDKitEncoder(nn.Module):
    # Encode RDKit features (Morgan, MACCS, RDKit_feats) into 3 tokens
    # Loads pre-computed features from npz file
    # Input: List of SMILES strings
    # Output: Dict with 3 tokens ['Morgan', 'MACCS', 'RDKit_feats']
    
    def __init__(self, 
                 rdkit_features_path: str,
                 morgan_dim: int = 1024, 
                 maccs_dim: int = 167, 
                 rdkit_desc_dim: int = 210):
        super().__init__()
        self.morgan_dim = morgan_dim
        self.maccs_dim = maccs_dim
        self.rdkit_desc_dim = rdkit_desc_dim
        
        # Load pre-computed RDKit features
        self.features_path = Path(rdkit_features_path)
        if not self.features_path.exists():
            raise FileNotFoundError(f"RDKit features file not found: {rdkit_features_path}")
        
        print(f"[RDKitEncoder] Loading features from: {rdkit_features_path}")
        data = np.load(self.features_path, allow_pickle=True)
        
        # FIX: Check for both 'SMILES' and 'smiles' keys
        if 'SMILES' in data:
            self.smiles_list = data['SMILES'].tolist()
        elif 'smiles' in data:
            self.smiles_list = data['smiles'].tolist()
        else:
            raise KeyError(f"Neither 'SMILES' nor 'smiles' found in {rdkit_features_path}. Available keys: {list(data.keys())}")
        
        # Extract features
        self.morgan_features = torch.from_numpy(data['morgan']).float()
        self.maccs_features = torch.from_numpy(data['maccs']).float()
        self.rdkit_features = torch.from_numpy(data['rdkit_feats']).float()
        
        # Create SMILES to index mapping for fast lookup
        self.smiles_to_idx = {smi: idx for idx, smi in enumerate(self.smiles_list)}
        
        print(f"[RDKitEncoder] Loaded {len(self.smiles_list)} molecules")
        print(f"[RDKitEncoder] Dims: Morgan={morgan_dim}, MACCS={maccs_dim}, RDKit={rdkit_desc_dim}")
    
    def forward(self, smiles: List[str]) -> Dict[str, torch.Tensor]:
        # Input: List of SMILES strings [B]
        # Output: Dict with 3 tokens
        #   'Morgan': [B, 1024]
        #   'MACCS': [B, 167]
        #   'RDKit_feats': [B, 210]
        
        batch_size = len(smiles)
        device = self.morgan_features.device
        
        # Initialize output tensors
        morgan_batch = torch.zeros(batch_size, self.morgan_dim, device=device)
        maccs_batch = torch.zeros(batch_size, self.maccs_dim, device=device)
        rdkit_batch = torch.zeros(batch_size, self.rdkit_desc_dim, device=device)
        
        # Track missing SMILES
        missing_count = 0
        
        # Look up features for each SMILES
        for i, smi in enumerate(smiles):
            if smi in self.smiles_to_idx:
                idx = self.smiles_to_idx[smi]
                morgan_batch[i] = self.morgan_features[idx]
                maccs_batch[i] = self.maccs_features[idx]
                rdkit_batch[i] = self.rdkit_features[idx]
            else:
                # SMILES not found - use zeros
                missing_count += 1
                if missing_count <= 3:  # Only print first 3 warnings
                    print(f"Warning: SMILES not found in RDKit features: {smi[:50]}...")
        
        if missing_count > 0:
            print(f"[RDKitEncoder] {missing_count}/{batch_size} SMILES not found in pre-computed features")
        
        # Clip extreme values to prevent Inf/NaN
        # Step 1: Replace NaN and Inf with finite values
        morgan_batch = torch.nan_to_num(morgan_batch, nan=0.0, posinf=1e6, neginf=-1e6)
        maccs_batch = torch.nan_to_num(maccs_batch, nan=0.0, posinf=1e6, neginf=-1e6)
        rdkit_batch = torch.nan_to_num(rdkit_batch, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Step 2: Additional clipping to prevent extreme values that cause BatchNorm issues
        # Clip to a reasonable range based on typical feature scales
        morgan_batch = torch.clamp(morgan_batch, min=-100.0, max=100.0)
        maccs_batch = torch.clamp(maccs_batch, min=-100.0, max=100.0)
        rdkit_batch = torch.clamp(rdkit_batch, min=-100.0, max=100.0)
        

        # Step 3: Use global statistics for normalization (computed once from training data)
        # This avoids batch-size dependency and ensures consistency
        # if not hasattr(self, 'global_stats_computed'):
        #     # Compute and cache global statistics on first call
        #     if batch_size > 1:
        #         eps = 1e-8
                
        #         # Cache statistics for future use (detach to avoid graph retention)
        #         self.morgan_mean = morgan_batch.mean(dim=0, keepdim=True).detach()
        #         self.morgan_std = torch.clamp(morgan_batch.std(dim=0, keepdim=True), min=eps).detach()
                
        #         self.maccs_mean = maccs_batch.mean(dim=0, keepdim=True).detach()
        #         self.maccs_std = torch.clamp(maccs_batch.std(dim=0, keepdim=True), min=eps).detach()
                
        #         self.rdkit_mean = rdkit_batch.mean(dim=0, keepdim=True).detach()
        #         self.rdkit_std = torch.clamp(rdkit_batch.std(dim=0, keepdim=True), min=eps).detach()
                
        #         self.global_stats_computed = True
        
        # # Apply cached normalization (works for any batch size including 1)
        # if hasattr(self, 'global_stats_computed'):
        #     morgan_batch = (morgan_batch - self.morgan_mean) / self.morgan_std
        #     maccs_batch = (maccs_batch - self.maccs_mean) / self.maccs_std
        #     rdkit_batch = (rdkit_batch - self.rdkit_mean) / self.rdkit_std
        
        return {
            'Morgan': morgan_batch,
            'MACCS': maccs_batch,
            'RDKit_feats': rdkit_batch
        }

    
    def get_output_dims(self) -> Tuple[int, int, int]:
        # Return output dimensions for each token
        return self.morgan_dim, self.maccs_dim, self.rdkit_desc_dim
    
    def to(self, device):
        # Override to move cached features to device
        super().to(device)
        
        # Move all cached feature tensors to device
        if hasattr(self, 'morgan_features'):
            self.morgan_features = self.morgan_features.to(device)
        if hasattr(self, 'maccs_features'):
            self.maccs_features = self.maccs_features.to(device)
        if hasattr(self, 'rdkit_features'):
            self.rdkit_features = self.rdkit_features.to(device)
        
        print(f"[RDKitEncoder] Successfully moved to device: {device}")
        return self
    
    def _apply(self, fn):
        # Override _apply to ensure cached features are also processed
        super()._apply(fn)
        
        # Apply function to cached features
        if hasattr(self, 'morgan_features'):
            self.morgan_features = fn(self.morgan_features)
        if hasattr(self, 'maccs_features'):
            self.maccs_features = fn(self.maccs_features)
        if hasattr(self, 'rdkit_features'):
            self.rdkit_features = fn(self.rdkit_features)
        
        return self