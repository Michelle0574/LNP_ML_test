#!/usr/bin/env python3
# Evaluate best attention models from hyperparameter search on all 5 folds
# Generates aggregated scores (test_scores_reg_agg.csv, test_scores_clf_agg.csv)
# Format is consistent with original Chemprop model for direct comparison

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
from models.trainer import LNPDataset, collate_fn
from models.predictor import LNPPredictor, aggregate_cv_scores
from models.data_preprocessing import preprocess_dataframe


def load_data_from_all_data(split_csv_path, all_data_path):
    # Load split CSV and merge with all_data.csv to get all features
    split_df = pd.read_csv(split_csv_path)
    all_data = pd.read_csv(all_data_path, low_memory=False)
    
    merged_df = split_df.merge(all_data, on='smiles', how='left', suffixes=('', '_alldata'))
    
    # Remove duplicate columns from all_data
    target_cols_to_drop = [col for col in merged_df.columns if col.endswith('_alldata')]
    if target_cols_to_drop:
        merged_df = merged_df.drop(columns=target_cols_to_drop)
    
    return merged_df


def prepare_data_for_eval(config, split_dir, fold_idx):
    # Load test data for evaluation
    split_path = Path(split_dir) / f'cv_{fold_idx}'
    all_data_path = Path('../data/all_data.csv')
    
    if not all_data_path.exists():
        raise FileNotFoundError(f"all_data.csv not found at {all_data_path}")
    
    print(f"\nLoading test data for fold {fold_idx}...")
    test_df = load_data_from_all_data(split_path / 'test.csv', all_data_path)
    
    # Load target roles
    target_roles_file = Path('../data/args_files/target_roles.json')
    with open(target_roles_file, 'r') as f:
        target_roles = json.load(f)
    
    reg_tasks = target_roles.get('regression', target_roles.get('regression_targets', []))
    clf_tasks = target_roles.get('classification', target_roles.get('classification_targets', []))
    
    config.task_names = reg_tasks + clf_tasks
    config.task_types = {}
    for task in reg_tasks:
        config.task_types[task] = 'regression'
    for task in clf_tasks:
        config.task_types[task] = 'classification'
    
    # Preprocess data (PDI, Purity, etc.)
    test_df = preprocess_dataframe(test_df, config)
    
    # ============== NEW: Align features with training data ==============
    # Load training data to detect which features actually exist in training set
    print(f"[Info] Detecting features from training data to ensure alignment...")
    train_df_sample = load_data_from_all_data(split_path / 'train.csv', all_data_path)
    train_df_sample = preprocess_dataframe(train_df_sample, config)
    
    # Get training data columns
    train_columns = set(train_df_sample.columns)
    test_columns = set(test_df.columns)
    
    # Find columns in test but not in train (these are the extra ones)
    extra_columns = test_columns - train_columns
    if extra_columns:
        print(f"[Warning] Found {len(extra_columns)} extra columns in test data (will be removed):")
        for col in sorted(extra_columns):
            print(f"  - {col}")
        # Remove extra columns from test data
        test_df = test_df.drop(columns=list(extra_columns))
    
    # Update data grouping from TRAINING DataFrame (not test!)
    config.update_data_grouping_from_dataframe(train_df_sample)
    # ====================================================================
    
    # Get feature columns
    feature_cols = {
        'comp': config.data_grouping['comp'],
        'phys': config.get_phys_cols_with_pdi(),
        'help': config.data_grouping['help'],
        'exp': config.data_grouping['exp']
    }
    
    print(f"[Info] Feature alignment completed:")
    print(f"  Composition features: {len(feature_cols['comp'])}")
    print(f"  Physical features: {len(feature_cols['phys'])}")
    print(f"  Helper lipid features: {len(feature_cols['help'])}")
    print(f"  Experimental features: {len(feature_cols['exp'])}")
    
    # Create dataset
    test_dataset = LNPDataset(test_df, config.smiles_column,
                             config.task_names, config.task_types, feature_cols)
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=0, pin_memory=True)
    
    return test_loader


def evaluate_fold(config, split_dir, fold_idx, model_path, output_dir, device):
    # Evaluate a single fold using pre-trained model
    
    print(f"\n{'='*80}")
    print(f"Evaluating Fold {fold_idx}")
    print(f"{'='*80}\n")
    
    # Prepare data
    test_loader = prepare_data_for_eval(config, split_dir, fold_idx)
    
    # Set MPNN checkpoint path for this fold
    use_regression = config.mpnn_encoder.get('use_regression_checkpoint', True)
    
    # Original checkpoint location (may not exist)
    if use_regression:
        mpnn_checkpoint = Path(split_dir) / f'cv_{fold_idx}' / 'fold_0' / 'model_0' / 'model.pt'
    else:
        clf_split_dir = Path(str(split_dir) + '_clf')
        mpnn_checkpoint = clf_split_dir / f'cv_{fold_idx}' / 'fold_0' / 'model_0' / 'model.pt'
    
    # Fallback: try to use checkpoint from by_source_smiles_with_ultra_held_out
    if not mpnn_checkpoint.exists():
        print(f"[Warning] Chemprop checkpoint not found at: {mpnn_checkpoint}")
        fallback_split_dir = '../data/crossval_splits/by_source_smiles_with_ultra_held_out'
        if use_regression:
            mpnn_checkpoint_fallback = Path(fallback_split_dir) / f'cv_{fold_idx}' / 'fold_0' / 'model_0' / 'model.pt'
        else:
            mpnn_checkpoint_fallback = Path(fallback_split_dir + '_clf') / f'cv_{fold_idx}' / 'fold_0' / 'model_0' / 'model.pt'
        
        if mpnn_checkpoint_fallback.exists():
            mpnn_checkpoint = mpnn_checkpoint_fallback
            print(f"[Info] Using fallback Chemprop checkpoint: {mpnn_checkpoint}")
        else:
            print(f"[Warning] Fallback checkpoint also not found at: {mpnn_checkpoint_fallback}")
            config.mpnn_checkpoint_path = None
            mpnn_checkpoint = None
    
    if mpnn_checkpoint and mpnn_checkpoint.exists():
        config.mpnn_checkpoint_path = str(mpnn_checkpoint)
        print(f"[Info] Using Chemprop checkpoint: {mpnn_checkpoint}")
    else:
        config.mpnn_checkpoint_path = None
    
    # Create model
    print(f"\nInitializing model...")
    model = LNPMixAttentionModel(config)
    
    # Load trained weights with strict=False to handle potential dimension mismatch
    print(f"Loading model weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model weights from checkpoint (handles both formats)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"[Info] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint
    
    # Use strict=False to allow partial loading (e.g., different exp feature dimensions)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"[Warning] Missing keys in checkpoint (will be randomly initialized): {len(missing_keys)} keys")
        if len(missing_keys) <= 10:
            for key in missing_keys:
                print(f"  - {key}")
        else:
            print(f"  First 10: {missing_keys[:10]}")
            
    if unexpected_keys:
        print(f"[Warning] Unexpected keys in checkpoint (will be ignored): {len(unexpected_keys)} keys")
        if len(unexpected_keys) <= 10:
            for key in unexpected_keys:
                print(f"  - {key}")
        else:
            print(f"  First 10: {unexpected_keys[:10]}")
    
    model = model.to(device)
    model.eval()
    
    # Create output directory
    fold_output = Path(output_dir) / f'fold_{fold_idx}'
    fold_output.mkdir(parents=True, exist_ok=True)
    
    # Evaluate
    print(f"\nEvaluating on test set...")
    predictor = LNPPredictor(model, config, device=device)
    predictions, targets, masks, smiles = predictor.predict(test_loader)
    
    # Compute and save scores
    predictor.compute_scores(predictions, targets, masks, fold_output)
    
    # Save predictions
    predictor.save_predictions(predictions, targets, masks, smiles,
                              fold_output / 'predictions.csv')
    
    print(f"Fold {fold_idx} evaluation completed!")
    print(f"Results saved to: {fold_output}\n")

def evaluate_all_folds(config, split_dir, model_path, output_dir, num_folds, device):
    # Evaluate all folds and aggregate scores
    
    print(f"\n{'='*80}")
    print(f"Evaluating Best Attention Model on All Folds")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Split: {split_dir}")
    print(f"Output: {output_dir}")
    print(f"Folds: {num_folds}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    # Evaluate each fold
    for fold_idx in range(num_folds):
        evaluate_fold(config, split_dir, fold_idx, model_path, output_dir, device)
    
    # Aggregate scores
    print(f"\n{'='*80}")
    print(f"Aggregating Cross-Validation Scores")
    print(f"{'='*80}\n")
    
    # Load target roles to get task types
    target_roles_file = Path('../data/args_files/target_roles.json')
    with open(target_roles_file, 'r') as f:
        target_roles = json.load(f)
    
    reg_tasks = target_roles.get('regression', target_roles.get('regression_targets', []))
    clf_tasks = target_roles.get('classification', target_roles.get('classification_targets', []))
    
    task_types = {}
    for task in reg_tasks:
        task_types[task] = 'regression'
    for task in clf_tasks:
        task_types[task] = 'classification'
    
    # Aggregate
    aggregate_cv_scores(output_dir, num_folds, task_types)
    
    print(f"\n{'='*80}")
    print(f"Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Aggregated scores saved to:")
    print(f"  - {output_dir}/test_scores_reg_agg.csv")
    print(f"  - {output_dir}/test_scores_clf_agg.csv")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate best attention models from hyperparameter search',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('model_type', choices=['regression', 'classification'],
                       help='Type of best model to evaluate')
    parser.add_argument('--split_name', type=str, default='by_source_smiles',
                       help='Name of the split')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--search_version', type=str, default='v1',
                        choices=['v1', 'v3', 'v5'],
                        help='Hyperparameter search version (v1, v3, or v5)')

    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Determine search directory and output suffix based on version
    if args.search_version == 'v5':
        search_dir = Path('../results/hyperparam_search_v5')
        suffix = '_v5_extended'
    elif args.search_version == 'v3':
        search_dir = Path('../results/hyperparam_search_v3')
        suffix = '_v3_extended'
    else:
        search_dir = Path('../results/hyperparam_search')
        suffix = '_v1_extended'

    if args.model_type == 'regression':
        config_path = search_dir / 'best_regression_config.json'
        model_path = search_dir / 'best_regression_model.pt'
        output_dir = Path(f'../results/attention_model_best_regression{suffix}/{args.split_name}')
    else:
        config_path = search_dir / 'best_classification_config.json'
        model_path = search_dir / 'best_classification_model.pt'
        output_dir = Path(f'../results/attention_model_best_classification{suffix}/{args.split_name}')
    
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        print(f"Please run hyperparameter search first.")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"Error: Model weights not found: {model_path}")
        print(f"Please run hyperparameter search first.")
        sys.exit(1)
    
    # Load config
    print(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))
    
    config.smiles_column = 'smiles'
    config.num_workers = 0
    
    # Set split directory
    split_dir = f'../data/crossval_splits/{args.split_name}'
    
    if not Path(split_dir).exists():
        print(f"Error: Split directory not found: {split_dir}")
        sys.exit(1)
    
    # Evaluate
    evaluate_all_folds(config, split_dir, str(model_path), str(output_dir), 
                      args.num_folds, device)
                    

if __name__ == '__main__':
    main()