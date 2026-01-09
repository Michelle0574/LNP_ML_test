import torch
import torch.nn as nn
from typing import List, Dict


class TabularEncoder(nn.Module):
    # Encode tabular features into 4 tokens: [comp, phys, help, exp]
    # Input: Dict[str, torch.Tensor] from DataLoader (features already grouped)
    # Output: Dict[str, torch.Tensor] with 4 tokens
    
    def __init__(self, comp_cols: List[str], phys_cols: List[str], 
                 help_prefixes: List[str], exp_prefixes: List[str]):
        super().__init__()
        self.comp_cols = comp_cols
        self.phys_cols = phys_cols
        self.help_prefixes = help_prefixes
        self.exp_prefixes = exp_prefixes
        
        print(f"[TabularEncoder] Initialized:")
        print(f"  Comp columns: {len(comp_cols)}")
        print(f"  Phys columns: {len(phys_cols)}")
        print(f"  Help prefixes: {help_prefixes}")
        print(f"  Exp prefixes: {exp_prefixes}")
    
    def forward(self, tabular_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Input: Dict with keys 'comp', 'phys', 'help', 'exp'
        #   Each value is a tensor [B, D_i] where D_i is the feature dimension
        # Output: Same dict (pass-through, features already grouped by DataLoader)
        
        # The DataLoader (trainer.py) already groups features correctly:
        # - 'comp': [B, 9] - composition features
        # - 'phys': [B, 9] - physical features (including processed PDI)
        # - 'help': [B, 4] - helper lipid one-hot features
        # - 'exp': [B, 20] - experimental condition one-hot features (including processed Purity)
        
        # Simply return the dict as-is
        # If we wanted to add learned transformations, we could add linear layers here
        return tabular_data
    
    def get_output_dims(self) -> Dict[str, int]:
        # Return expected output dimensions for each token type
        # Note: Actual dimensions are determined by DataLoader based on available columns
        return {
            'comp': len(self.comp_cols),
            'phys': len(self.phys_cols),
            'help': 'variable',  # Depends on one-hot encoding
            'exp': 'variable'    # Depends on one-hot encoding
        }