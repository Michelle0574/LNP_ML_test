import json
import os
from typing import Dict, Any, List


class AttentionConfig:
    # Global configuration for attention-based LNP model
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                '../../data/args_files/attention_config.json'
            )
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Main hyperparameters
        self.model_name = self.config.get('model_name', 'LNPMix_Attention')
        self.d_model = self.config['d_model']
        self.num_heads = self.config['num_heads']
        self.dropout = self.config['dropout']
        
        # Validate
        assert self.d_model % self.num_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        
        self.d_head = self.d_model // self.num_heads
        
        # MPNN encoder config
        self.mpnn_encoder = self.config.get('mpnn_encoder', {})
        
        # Projection config
        self.projection = self.config['projection']
        
        # Cross-attention config
        self.cross_attention = self.config['cross_attention']
        
        # Channel join config
        self.channel_join = self.config['channel_join']
        
        # Pooling config
        self.pooling = self.config['pooling']
        
        # Task heads config
        self.heads = self.config['heads']
        
        # Ablation switches
        self.ablation = self.config['ablation_switches']
        
        # Data groups
        self.data_groups = self.config['data_groups']
        
        # RDKit features config
        self.rdkit_features = self.config['rdkit_features']

        # Training config
        self.training = self.config.get('training', {})
        
        # Add convenience properties for model usage
        self._setup_convenience_properties()
        
        # Token weighting and gradient clipping configuration
        # NOTE: Must be after self.training is set
        self.use_token_weights = self.config.get('use_token_weights', True)
        self.gradient_clip_norm = self.training.get('gradient_clip_norm', 1.0)
        
        # Task-specific dropouts (NEW)
        self.classification_dropout = self.config.get('classification_dropout', self.dropout)
        self.regression_dropout = self.config.get('regression_dropout', self.dropout)
        
        # Token weight initialization strategy (NEW)
        self.token_weight_init_strategy = self.config.get('token_weight_init_strategy', 'balanced')
        
        # Task loss weighting (NEW)
        self.use_task_loss_weights = self.config.get('use_task_loss_weights', True)
        
        # Label smoothing (NEW)
        self.label_smoothing = self.config.get('heads', {}).get('classification', {}).get('label_smoothing', 0.0)
        
        print(f"[Config] Token weights: {'Enabled' if self.use_token_weights else 'Disabled'}")
        print(f"[Config] Gradient clip norm: {self.gradient_clip_norm}")
        print(f"[Config] Classification dropout: {self.classification_dropout:.3f}")
        print(f"[Config] Regression dropout: {self.regression_dropout:.3f}")
        print(f"[Config] Token init strategy: {self.token_weight_init_strategy}")
        print(f"[Config] Task loss weighting: {self.use_task_loss_weights}")
        print(f"[Config] Label smoothing: {self.label_smoothing}")

    def _setup_convenience_properties(self):
        # Setup convenient access properties based on config structure
        
        # MPNN properties
        self.mpnn_checkpoint_path = self.mpnn_encoder.get('checkpoint_path')
        self.mpnn_freeze = self.mpnn_encoder.get('freeze', False)
        self.mpnn_output_dim = self.mpnn_encoder.get('output_dim', 300)
        
        # Projection properties
        self.projector_hidden_dims = self.projection.get('hidden_dims', [])
        
        # Attention properties
        self.num_attention_layers = self.cross_attention.get('layers', 1)
        
        # Ablation properties
        self.disable_mpnn_token = self.ablation.get('disable_mpnn_token', False)
        
        # RDKit feature properties
        self.rdkit_features_path = self.rdkit_features['path']
        self.morgan_dim = self.rdkit_features['morgan_dim']
        self.maccs_dim = self.rdkit_features['maccs_dim']
        self.rdkit_feats_dim = self.rdkit_features['rdkit_desc_dim']
        
        # Training properties
        self.batch_size = self.training.get('batch_size', 32)
        self.learning_rate = self.training.get('learning_rate', 0.001)
        self.weight_decay = self.training.get('weight_decay', 1e-5)
        self.early_stopping_patience = self.training.get('early_stopping_patience', 10)
        self.save_frequency = self.training.get('save_frequency', 10)
        
        # Head properties
        self.head_hidden_dims = self.heads['regression'].get('hidden_dims', [256, 128])
        
        # Fusion type (for compatibility)
        self.fusion_type = 'mean'  # Default fusion type
        
        # Task weights (for multi-task loss)
        self.task_weights = {}
        
        # Data grouping - build feature column lists from prefixes
        self.data_grouping = self._build_data_grouping()
        
        # FIX: Set preprocessing flags here so they're available immediately
        self.pdi_enabled = self.data_groups.get('pdi_processing', {}).get('enabled', False)
        self.purity_enabled = self.data_groups.get('purity_processing', {}).get('enabled', False)
    
    def _build_data_grouping(self) -> Dict[str, List[str]]:
        # Build data grouping dict compatible with TabularEncoder
        # Start with explicit columns from config
        
        return {
            'comp': self.data_groups.get('comp_cols', []),
            'phys': self.data_groups.get('phys_cols', []),
            'help': [],  # Will be populated from prefixes
            'exp': []    # Will be populated from prefixes
        }
    
    def update_data_grouping_from_dataframe(self, df_columns: List[str]):
        # Update help and exp columns based on actual dataframe columns
        # Also handles PDI processing
        
        # Collect one-hot encoded columns
        help_prefixes = self.data_groups.get('help_prefixes', [])
        exp_prefixes = self.data_groups.get('exp_prefixes', [])
        phys_prefixes = self.data_groups.get('phys_prefixes', [])  
        extra_x_categorical = ['Delivery_target', 'Helper_lipid_ID', 'Route_of_administration', 
                    'Batch_or_individual_or_barcoded', 'Cargo_type', 'Model_type',
                    'Purity', 'Mix_type', 'Target_or_delivered_gene', 'Value_name']
        
        help_cols = []
        exp_cols = []
        phys_cols_from_prefix = []  
        
        if help_prefixes:
            for prefix in help_prefixes:
                help_cols.extend([col for col in df_columns if col.startswith(prefix)])
        
        if exp_prefixes:
            for prefix in exp_prefixes:
                exp_cols.extend([col for col in df_columns if col.startswith(prefix)])
        
        # Collect phys columns from prefixes (for Target_or_delivered_gene, etc.)
        if phys_prefixes:
            for prefix in phys_prefixes:
                phys_cols_from_prefix.extend([col for col in df_columns if col.startswith(prefix)])
        
        # Filter to only include columns that actually exist in the data
        help_cols = [c for c in help_cols if c in df_columns]
        exp_cols = [c for c in exp_cols if c in df_columns]
        phys_cols_from_prefix = [c for c in phys_cols_from_prefix if c in df_columns]
        
        # Merge phys_cols from config with dynamically detected ones
        existing_phys = self.data_groups.get('phys_cols', [])
        all_phys_cols = sorted(set(existing_phys + phys_cols_from_prefix))
        
        print(f"\n[Config] Feature detection:")
        print(f"  Helper lipid (one-hot): {len(help_cols)} columns")
        print(f"  Physical (one-hot): {len(all_phys_cols)} columns")
        print(f"  Experimental (one-hot): {len(exp_cols)} columns")
        
        # Store for reference
        self.data_grouping['help'] = sorted(set(help_cols))
        self.data_grouping['phys'] = all_phys_cols  # UPDATED to include dynamic columns
        self.data_grouping['exp'] = sorted(set(exp_cols))
        
        return self.data_grouping
    
    def get_phys_cols_with_pdi(self):
        # Get physical feature columns including processed PDI
        phys_cols = list(self.data_groups.get('phys_cols', []))
        
        if self.pdi_enabled:
            # Replace PDI with processed columns
            if 'PDI' in phys_cols:
                phys_cols.remove('PDI')
            pdi_output = self.data_groups.get('pdi_processing', {}).get('output_cols', [])
            phys_cols.extend(pdi_output)
        
        return phys_cols
    
    
    def get(self, key: str, default: Any = None) -> Any:
        # Get config value by key path (e.g., 'cross_attention.layers')
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def summary(self) -> str:
        # Print configuration summary
        lines = [
            "="*60,
            f"Model: {self.model_name}",
            "="*60,
            f"d_model: {self.d_model}",
            f"num_heads: {self.num_heads}",
            f"d_head: {self.d_head}",
            f"dropout: {self.dropout}",
            "",
            "MPNN Encoder:",
            f"  checkpoint: {self.mpnn_encoder.get('checkpoint_path', 'None')}",
            f"  freeze: {self.mpnn_encoder.get('freeze', False)}",
            f"  output_dim: {self.mpnn_encoder.get('output_dim', 300)}",
            "",
            "Ablation Switches (all tokens enabled by default):",
            f"  disable_rdkit_tokens: {self.ablation.get('disable_rdkit_tokens', False)}",
            f"  disable_mpnn_token: {self.ablation.get('disable_mpnn_token', False)}",
            f"  disable_tabular_tokens: {self.ablation.get('disable_tabular_tokens', False)}",
            f"  disable_cross_attention: {self.ablation.get('disable_cross_attention', False)}",
            f"  disable_pooling: {self.ablation.get('disable_pooling', False)}",
            "",
            "Cross-Attention:",
            f"  enabled: {self.cross_attention['enabled']}",
            f"  layers: {self.cross_attention['layers']}",
            f"  bi_directional: {self.cross_attention['bi_directional']}",
            "",
            "Pooling:",
            f"  enabled: {self.pooling['enabled']}",
            f"  type: {self.pooling['type']}",
            "",
            "Training:",
            f"  batch_size: {self.training.get('batch_size', 32)}",
            f"  learning_rate: {self.training.get('learning_rate', 0.001)}",
            f"  weight_decay: {self.training.get('weight_decay', 1e-5)}",
            f"  early_stopping_patience: {self.training.get('early_stopping_patience', 10)}",
            f"  gradient_clip_norm: {self.gradient_clip_norm}",
            f"  loss_weights: {self.training.get('loss_weights', {})}",
            "",
            "Task-Specific Configuration:",
            f"  classification_dropout: {self.classification_dropout}",
            f"  regression_dropout: {self.regression_dropout}",
            f"  label_smoothing: {self.label_smoothing}",
            f"  use_task_loss_weights: {self.use_task_loss_weights}",
            "",
            "Token Weighting:",
            f"  use_token_weights: {self.use_token_weights}",
            f"  init_strategy: {self.token_weight_init_strategy}",
            "="*60
        ]
        return "\n".join(lines)


def load_config(config_path: str = None) -> AttentionConfig:
    # Factory function to load configuration
    return AttentionConfig(config_path)


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print(config.summary())
    print("\nRDKit features path:", config.rdkit_features['path'])
    print("Comp columns:", config.data_groups['comp_cols'][:3], "...")
    print("\nToken configuration:")
    print(f"  8 tokens will be used: [chem(MPNN), Morgan, MACCS, RDKit_feats, comp, phys, help, exp]")
    print(f"  All tokens enabled (ablation switches are for experiments only)")