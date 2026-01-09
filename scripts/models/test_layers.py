import torch
import sys
sys.path.append('..')
from models.config import load_config
from models.layers.token_projector import TokenProjector
from models.layers.cross_modal_attention import CrossModalAttention

def test_token_projector():
    print("\n" + "="*60)
    print("Testing TokenProjector")
    print("="*60)
    
    config = load_config()
    
    # Define input dimensions for 8 tokens
    input_dims = {
        'chem': 300,         # MPNN output
        'Morgan': 1024,      # Morgan fingerprint
        'MACCS': 167,        # MACCS keys
        'RDKit_feats': 210,  # RDKit descriptors
        'comp': 9,           # Composition features
        'phys': 7,           # Physical features
        'help': 50,          # Helper lipid one-hot (example)
        'exp': 30            # Experimental conditions one-hot (example)
    }
    
    projector = TokenProjector(input_dims, d_model=128, independent=True)
    
    # Create dummy input
    batch_size = 4
    token_dict = {
        'chem': torch.randn(batch_size, 300),
        'Morgan': torch.randn(batch_size, 1024),
        'MACCS': torch.randn(batch_size, 167),
        'RDKit_feats': torch.randn(batch_size, 210),
        'comp': torch.randn(batch_size, 9),
        'phys': torch.randn(batch_size, 7),
        'help': torch.randn(batch_size, 50),
        'exp': torch.randn(batch_size, 30)
    }
    
    # Forward pass
    output = projector(token_dict)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: [B=4, 8, d_model=128]")
    assert output.shape == (batch_size, 8, 128), f"Shape mismatch! Got {output.shape}"
    print("✓ TokenProjector test passed!")


def test_cross_attention():
    print("\n" + "="*60)
    print("Testing CrossModalAttention")
    print("="*60)
    
    batch_size = 4
    d_model = 128
    num_heads = 4
    
    attn = CrossModalAttention(d_model, num_heads, dropout=0.1)
    
    # Create dummy channels
    chan_a = torch.randn(batch_size, 4, d_model)  # [Morgan, MACCS, RDKit_feats, chem]
    chan_b = torch.randn(batch_size, 4, d_model)  # [comp, phys, help, exp]
    
    # Forward pass
    output, shapes = attn(chan_a, chan_b)
    
    print(f"\nInput chan_a shape: {chan_a.shape}")
    print(f"Input chan_b shape: {chan_b.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nIntermediate shapes:")
    for key, shape in shapes.items():
        print(f"  {key}: {shape}")
    
    assert output.shape == (batch_size, 4, d_model), f"Shape mismatch! Got {output.shape}"
    print("\n✓ CrossModalAttention test passed!")


if __name__ == "__main__":
    test_token_projector()
    test_cross_attention()
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)