import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from chemprop.utils import load_checkpoint


class MPNNEncoder(nn.Module):
    # Extract molecular embeddings from pre-trained Chemprop model
    # Input: List of SMILES strings
    # Output: Tensor[B, D_mpnn] where D_mpnn is the hidden dimension from Chemprop
    
    def __init__(self, checkpoint_path: str, freeze: bool = False, output_dim: int = 300):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.freeze = freeze
        self._configured_output_dim = output_dim  # Store configured dim
        self.actual_output_dim = None  # Will be set after first forward pass
        self.device = torch.device('cpu')  # Will be updated by to()
        
        print(f"[MPNNEncoder] Loading Chemprop model from: {checkpoint_path}")
        self._load_chemprop_model(checkpoint_path)
        
        # Try to infer output dimension from model
        self._infer_output_dim()
        
        if self.freeze:
            for param in self.chemprop_model.parameters():
                param.requires_grad = False
            print(f"[MPNNEncoder] Model frozen (no gradient updates)")
        else:
            print(f"[MPNNEncoder] Model trainable (will be jointly trained)")
    
    def _load_chemprop_model(self, checkpoint_path: str):
        # Load pre-trained Chemprop model
        try:
            # Use chemprop's built-in checkpoint loader
            self.chemprop_model = load_checkpoint(checkpoint_path, device=self.device)
            
            # Verify model structure
            if not hasattr(self.chemprop_model, 'encoder'):
                raise AttributeError("Loaded model does not have 'encoder' attribute")
            
            print(f"[MPNNEncoder] Successfully loaded Chemprop model")
            print(f"[MPNNEncoder] Model type: {type(self.chemprop_model).__name__}")
            
            # Check if model uses input features
            self.use_input_features = self.chemprop_model.encoder.use_input_features
            if self.use_input_features:
                if hasattr(self.chemprop_model, 'args') and hasattr(self.chemprop_model.args, 'features_size'):
                    self.features_size = self.chemprop_model.args.features_size
                    print(f"[MPNNEncoder] Model uses input features with size: {self.features_size}")
                else:
                    self.features_size = 1
                    print(f"[MPNNEncoder] Warning: Could not get features_size, defaulting to 1")
            else:
                self.features_size = 0
                print(f"[MPNNEncoder] Model does not use input features")
            
        except Exception as e:
            print(f"[MPNNEncoder] Error loading checkpoint: {e}")
            raise RuntimeError(f"Failed to load Chemprop model from {checkpoint_path}: {e}")
    
    def _infer_output_dim(self):
        # Try to infer actual output dimension from model architecture
        try:
            # Test with a dummy SMILES to get actual output dimension
            dummy_smiles = ['C']  # Simplest valid SMILES
            batch = [[s] for s in dummy_smiles]
            
            if self.use_input_features:
                features_batch = [np.zeros(self.features_size, dtype=np.float32)]
            else:
                features_batch = None
            
            with torch.no_grad():
                dummy_output = self.chemprop_model.fingerprint(
                    batch=batch,
                    features_batch=features_batch,
                    fingerprint_type='MPN'
                )
                self.actual_output_dim = dummy_output.shape[1]
            
            print(f"[MPNNEncoder] Detected actual output dimension: {self.actual_output_dim}")
            if self.actual_output_dim != self._configured_output_dim:
                print(f"[MPNNEncoder] WARNING: Configured dim ({self._configured_output_dim}) != Actual dim ({self.actual_output_dim})")
                print(f"[MPNNEncoder] Using actual dimension: {self.actual_output_dim}")
        
        except Exception as e:
            print(f"[MPNNEncoder] Could not infer output dimension: {e}")
            print(f"[MPNNEncoder] Will use configured dimension: {self._configured_output_dim}")
            self.actual_output_dim = self._configured_output_dim
    
    def forward(self, smiles_list: List[str]) -> torch.Tensor:
        # Extract MPNN embeddings for a batch of SMILES
        # Args:
        #     smiles_list: List[str], batch of SMILES strings
        # Returns:
        #     embeddings: Tensor[B, output_dim] on self.device
        
        if self.chemprop_model is None:
            raise RuntimeError("Chemprop model not initialized.")
        
        batch_size = len(smiles_list)
        
        # Convert SMILES to the format expected by Chemprop encoder
        batch = [[s] for s in smiles_list]
        
        # Prepare features_batch
        if self.use_input_features:
            features_batch = [np.zeros(self.features_size, dtype=np.float32) for _ in range(batch_size)]
        else:
            features_batch = None
        
        # Extract embeddings - Chemprop will handle device internally
        if self.freeze:
            with torch.no_grad():
                embeddings = self.chemprop_model.fingerprint(
                    batch=batch,
                    features_batch=features_batch,
                    fingerprint_type='MPN'
                )
        else:
            embeddings = self.chemprop_model.fingerprint(
                batch=batch,
                features_batch=features_batch,
                fingerprint_type='MPN'
            )
        

        # Ensure output is on the correct device
        embeddings = embeddings.to(self.device)
        
        # Add safety preprocessing (same as RDKitEncoder)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=1e6, neginf=-1e6)
        embeddings = torch.clamp(embeddings, min=-100.0, max=100.0)
        
        # No normalization here - let TokenProjector's BatchNorm handle it
        # This avoids instability from first-batch statistics
        
        return embeddings

    
    def get_output_dim(self) -> int:
        # Return the actual output dimension of MPNN embeddings
        return self.actual_output_dim if self.actual_output_dim is not None else self._configured_output_dim
    
    def to(self, device):
        # Properly move model to device
        super().to(device)
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        
        if self.chemprop_model is not None:
            # Move chemprop model to device
            self.chemprop_model = self.chemprop_model.to(self.device)
            
            # Critical: Update the device attribute in encoder
            if hasattr(self.chemprop_model, 'encoder'):
                self.chemprop_model.encoder.device = self.device
                
                # Update device in nested encoders (ModuleList)
                if hasattr(self.chemprop_model.encoder, 'encoder'):
                    if isinstance(self.chemprop_model.encoder.encoder, nn.ModuleList):
                        for enc in self.chemprop_model.encoder.encoder:
                            if hasattr(enc, 'device'):
                                enc.device = self.device
                    elif hasattr(self.chemprop_model.encoder.encoder, 'device'):
                        self.chemprop_model.encoder.encoder.device = self.device
            
            print(f"[MPNNEncoder] Successfully moved to device: {self.device}")
        
        return self
    
    def _apply(self, fn):
        # Override _apply to ensure chemprop_model is also processed
        super()._apply(fn)
        
        if self.chemprop_model is not None:
            self.chemprop_model = self.chemprop_model._apply(fn)
            
            # Update device attribute after _apply
            if hasattr(self.chemprop_model, 'encoder'):
                # Infer device from first parameter
                first_param = next(self.chemprop_model.parameters(), None)
                if first_param is not None:
                    inferred_device = first_param.device
                    self.chemprop_model.encoder.device = inferred_device
                    self.device = inferred_device
                    
                    # Update nested encoders
                    if hasattr(self.chemprop_model.encoder, 'encoder'):
                        if isinstance(self.chemprop_model.encoder.encoder, nn.ModuleList):
                            for enc in self.chemprop_model.encoder.encoder:
                                if hasattr(enc, 'device'):
                                    enc.device = inferred_device
        
        return self
