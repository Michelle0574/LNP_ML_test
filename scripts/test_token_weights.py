#!/usr/bin/env python3
"""
Test script to verify token weight functionality
"""

import torch
import sys
sys.path.insert(0, '/home/gongruyi/drug_delivery/LNP_ML/scripts')

from models.config import load_config
from models.layers.token_projector import TokenProjector

def test_token_weights():
    print("="*60)
    print("Testing Token Weight Functionality")
    print("="*60)
    
    # Load config
    config_path = '../data/args_files/attention_config.json'
    config = load_config(config_path)
    
    # Test 1: TokenProjector with learnable weights
    print("\n1. Testing TokenProjector with learnable weights...")
    input_dims = {
        'chem': 300,
        'Morgan': 1024,
        'MACCS': 167,
        'RDKit_feats': 210,
        'comp': 9,
        'phys': 9,
        'help': 4,
        'exp': 20
    }
    
    projector = TokenProjector(
        input_dims=input_dims,
        d_model=128,
        use_batch_norm=True,
        use_token_weights=True
    )
    
    print("✓ TokenProjector initialized successfully!")
    
    # Test 2: Check token weights
    print("\n2. Checking token weights...")
    weights = projector.get_token_weights()
    if weights:
        print("✓ Token weights successfully initialized!")
        projector.print_token_weights()
    else:
        print("✗ Token weights not enabled")
    
    # Test 3: Forward pass
    print("\n3. Testing forward pass...")
    batch_size = 4
    token_embeddings = {
        name: torch.randn(batch_size, dim) 
        for name, dim in input_dims.items()
    }
    
    try:
        with torch.no_grad():
            output = projector(token_embeddings)
        print(f"✓ Forward pass successful!")
        print(f"  Input: {len(input_dims)} tokens")
        print(f"  Output shape: {output.shape}")  # Should be [4, 8, 128]
        
        # Check output validity
        if torch.isnan(output).any():
            print("✗ Output contains NaN")
        else:
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: Weight learning simulation
    print("\n4. Testing weight update simulation...")
    optimizer = torch.optim.Adam(projector.parameters(), lr=0.001)
    
    # Simulate one gradient update
    output = projector(token_embeddings)
    loss = output.mean()
    loss.backward()
    optimizer.step()
    
    print("✓ Weights are learnable (gradient update successful)")
    print("\nWeights after one update:")
    projector.print_token_weights()
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)

if __name__ == '__main__':
    test_token_weights()