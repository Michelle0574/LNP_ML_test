#!/usr/bin/env python3
# Training script for LNPMix Attention Model
# Integrates with existing cross-validation splits from main_script.py

import sys
import argparse
import torch
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import DataLoader

# Import attention model modules
from models.config import load_config
from models.lnpmix_model import LNPMixAttentionModel
from models.trainer import LNPDataset, LNPTrainer, collate_fn
from models.predictor import LNPPredictor, aggregate_cv_scores
from models.data_preprocessing import preprocess_dataframe


def load_data_from_all_data(split_csv_path, all_data_path):
    # Load split CSV (only has SMILES + targets) and merge with all_data.csv
    # to get all features
    
    split_df = pd.read_csv(split_csv_path)
    all_data = pd.read_csv(all_data_path, low_memory=False)
    
    # Merge on SMILES to get all features
    # Use left join to keep all samples from split
    merged_df = split_df.merge(all_data, on='smiles', how='left', suffixes=('', '_alldata'))
    
    # For duplicate columns, prefer the split version (targets), use all_data version for features
    # Remove duplicate target columns from all_data
    target_cols_to_drop = [col for col in merged_df.columns if col.endswith('_alldata')]
    if target_cols_to_drop:
        merged_df = merged_df.drop(columns=target_cols_to_drop)
    
    return merged_df


def prepare_data(config, split_dir, fold_idx):
    # Load train/valid/test data for a specific fold
    Load from all_data.csv instead of extra_x.csv
    
    split_path = Path(split_dir) / f'cv_{fold_idx}'
    all_data_path = Path('../data/all_data.csv')
    
    if not all_data_path.exists():
        raise FileNotFoundError(f"all_data.csv not found at {all_data_path}")
    
    print(f"\nLoading data from all_data.csv...")
    
    # Load data by merging split CSVs with all_data.csv
    train_df = load_data_from_all_data(split_path / 'train.csv', all_data_path)
    valid_df = load_data_from_all_data(split_path / 'valid.csv', all_data_path)
    test_df = load_data_from_all_data(split_path / 'test.csv', all_data_path)
    
    print(f"\nFold {fold_idx} data loaded:")
    print(f"  Train: {len(train_df)} samples, {len(train_df.columns)} columns")
    print(f"  Valid: {len(valid_df)} samples, {len(valid_df.columns)} columns")
    print(f"  Test:  {len(test_df)} samples, {len(test_df.columns)} columns")
    
    # Load target_roles.json to get task info
    target_roles_file = Path('../data/args_files/target_roles.json')
    
    with open(target_roles_file, 'r') as f:
        target_roles = json.load(f)
    
    # Update config with task information
    reg_tasks = target_roles.get('regression', target_roles.get('regression_targets', []))
    clf_tasks = target_roles.get('classification', target_roles.get('classification_targets', []))
    
    config.task_names = reg_tasks + clf_tasks
    config.task_types = {}
    for task in reg_tasks:
        config.task_types[task] = 'regression'
    for task in clf_tasks:
        config.task_types[task] = 'classification'
    
    print(f"\nTasks: {len(config.task_names)} total")
    print(f"  Regression: {len(reg_tasks)}")
    print(f"  Classification: {len(clf_tasks)}")
    
    # FIX: Preprocess FIRST to generate Purity_Pure and Purity_Crude columns
    print("\nPreprocessing data...")
    train_df = preprocess_dataframe(train_df, config)
    valid_df = preprocess_dataframe(valid_df, config)
    test_df = preprocess_dataframe(test_df, config)
    
    # THEN update data_grouping (now Purity_ columns exist)
    all_columns = train_df.columns.tolist()
    config.update_data_grouping_from_dataframe(all_columns)
    
    # Get feature columns from updated data_grouping
    feature_cols = {
        'comp': config.data_grouping['comp'],
        'phys': config.get_phys_cols_with_pdi(),
        'help': config.data_grouping['help'],
        'exp': config.data_grouping['exp']
    }
    
    # Verify all feature columns exist in data
    all_feature_cols = (feature_cols['comp'] + feature_cols['phys'] + 
                       feature_cols['help'] + feature_cols['exp'])
    missing_cols = [col for col in all_feature_cols if col not in train_df.columns and not col.startswith('_dummy_')]
    
    if missing_cols:
        print(f"\n[Warning] Missing columns in data: {missing_cols[:10]}...")
        print(f"  These columns will be filled with zeros")
    
    print(f"\nFeature groups (after preprocessing):")
    print(f"  comp: {len(feature_cols['comp'])} features")
    print(f"  phys: {len(feature_cols['phys'])} features - {feature_cols['phys']}")
    print(f"  help: {len(feature_cols['help'])} features")
    if len(feature_cols['help']) <= 10:
        print(f"    {feature_cols['help']}")
    print(f"  exp: {len(feature_cols['exp'])} features")
    if len(feature_cols['exp']) <= 25:
        print(f"    {feature_cols['exp']}")
    else:
        print(f"    {feature_cols['exp'][:10]}... (+{len(feature_cols['exp'])-10} more)")
    
    # Create datasets
    train_dataset = LNPDataset(train_df, config.smiles_column, 
                              config.task_names, config.task_types, feature_cols)
    valid_dataset = LNPDataset(valid_df, config.smiles_column,
                              config.task_names, config.task_types, feature_cols)
    test_dataset = LNPDataset(test_df, config.smiles_column,
                             config.task_names, config.task_types, feature_cols)
    
    # Create dataloaders with num_workers=0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                             shuffle=True, collate_fn=collate_fn, 
                             num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=collate_fn, 
                             num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn, 
                            num_workers=0, pin_memory=True)
    
    return train_loader, valid_loader, test_loader


def train_fold(config, split_dir, fold_idx, output_dir, epochs, device):
    # Train attention model for a single fold
    
    print(f"\n{'='*80}")
    print(f"Training Attention Model - Fold {fold_idx}")
    print(f"{'='*80}\n")
    
    # Prepare data
    train_loader, valid_loader, test_loader = prepare_data(config, split_dir, fold_idx)
    
    # Use MPNN checkpoint: prefer config file setting, fallback to split directory
    if config.mpnn_checkpoint_path and Path(config.mpnn_checkpoint_path).exists():
        print(f"\n[Info] Using Chemprop checkpoint from config: {config.mpnn_checkpoint_path}")
    else:
        # Try to find checkpoint in the current split directory
        mpnn_checkpoint = Path(split_dir) / f'cv_{fold_idx}' / 'fold_0' / 'model_0' / 'model.pt'
        
        if mpnn_checkpoint.exists():
            print(f"\n[Info] Found Chemprop checkpoint in split dir: {mpnn_checkpoint}")
            config.mpnn_checkpoint_path = str(mpnn_checkpoint)
        elif config.mpnn_checkpoint_path:
            print(f"\n[Warning] Config checkpoint not found: {config.mpnn_checkpoint_path}")
            config.mpnn_checkpoint_path = None
        else:
            print(f"\n[Warning] No Chemprop checkpoint available")
            config.mpnn_checkpoint_path = None
            
    # Create model
    print("\nInitializing LNPMix Attention Model...")
    model = LNPMixAttentionModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create fold output directory
    fold_output = Path(output_dir) / f'fold_{fold_idx}'
    fold_output.mkdir(parents=True, exist_ok=True)
    
    # Train
    trainer = LNPTrainer(model, config, save_dir=fold_output, device=device)
    trainer.train(train_loader, valid_loader, epochs=epochs)
    
    # Load best model for prediction
    best_model_path = fold_output / 'best_model.pt'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nEvaluating on test set...")
    # Predict on test set
    predictor = LNPPredictor(model, config, device=device)
    predictions, targets, masks, smiles = predictor.predict(test_loader)
    
    # Compute and save scores (outputs test_scores.csv and test_scores_clf.csv)
    predictor.compute_scores(predictions, targets, masks, fold_output)
    
    # Save detailed predictions
    predictor.save_predictions(predictions, targets, masks, smiles,
                              fold_output / 'predictions.csv')
    
    print(f"\nFold {fold_idx} training completed!")
    print(f"Results saved to: {fold_output}")


def train_cross_validation(config, split_dir, output_dir, epochs, num_folds, device):
    # Train all folds in cross-validation
    
    print(f"\n{'='*80}")
    print(f"LNPMix Attention Model - Cross-Validation Training")
    print(f"{'='*80}")
    print(f"Split directory: {split_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of folds: {num_folds}")
    print(f"Epochs per fold: {epochs}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Train each fold
    for fold_idx in range(num_folds):
        train_fold(config, split_dir, fold_idx, output_dir, epochs, device)
    
    # Aggregate scores across folds
    print(f"\n{'='*80}")
    print(f"Aggregating Cross-Validation Scores")
    print(f"{'='*80}\n")
    
    # Load target roles to get task types
    target_roles_file = Path('../data/args_files/target_roles.json')
    
    with open(target_roles_file, 'r') as f:
        target_roles = json.load(f)
    
    # Handle both old and new key names
    reg_tasks = target_roles.get('regression', target_roles.get('regression_targets', []))
    clf_tasks = target_roles.get('classification', target_roles.get('classification_targets', []))
    
    task_types = {}
    for task in reg_tasks:
        task_types[task] = 'regression'
    for task in clf_tasks:
        task_types[task] = 'classification'
    
    # Aggregate scores (outputs test_scores_reg_agg.csv and test_scores_clf_agg.csv)
    aggregate_cv_scores(output_dir, num_folds, task_types)
    
    print(f"\n{'='*80}")
    print(f"Cross-validation training completed!")
    print(f"Aggregated scores saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train LNPMix Attention Model on cross-validation splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('split_name', type=str,
                       help='Name of the split (e.g., by_source_smiles)')
    
    # Optional arguments
    parser.add_argument('--config', type=str, 
                       default='../data/args_files/attention_config.json',
                       help='Path to attention model configuration file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs per fold')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--fold', type=int, default=None,
                       help='Train specific fold only (0-indexed), or all folds if not specified')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--output_dir', type=str, default=None,
                   help='Custom output directory')
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Set data paths
    config.smiles_column = 'smiles'
    config.num_workers = 0  # Avoid multiprocessing issues
    
    # Set split and output directories
    split_dir = f'../data/crossval_splits/{args.split_name}'
    output_dir = f'../results/attention_model/{args.split_name}'
    
    # Verify split directory exists
    if not Path(split_dir).exists():
        print(f"Error: Split directory not found: {split_dir}")
        print(f"Please run main_script.py split first to generate cross-validation splits.")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'../results/attention_model/{args.split_name}'

    # Train
    if args.fold is not None:
        # Train single fold
        train_fold(config, split_dir, args.fold, output_dir, args.epochs, device)
    else:
        # Train all folds
        train_cross_validation(config, split_dir, output_dir, 
                             args.epochs, args.num_folds, device)


if __name__ == '__main__':
    main()