#!/usr/bin/env python3
"""
Evaluate original Chemprop models with extended metrics
Re-runs predictions using the trained Chemprop models
Aligns with the evaluation approach of v1 and v3 attention models
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import torch
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, explained_variance_score
)
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

# Import Chemprop modules
try:
    from chemprop.train import predict
    from chemprop.data import MoleculeDataLoader, MoleculeDataset
    from chemprop.data.utils import get_data, get_data_from_smiles
    from chemprop.utils import load_args, load_checkpoint, load_scalers
    import chemprop
except ImportError:
    print("Error: chemprop not installed. Please install it first.")
    sys.exit(1)


def sigmoid(x):
    """Sigmoid function to convert logits to probabilities"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def compute_extended_regression_metrics(y_true, y_pred):
    """Compute extended regression metrics"""
    metrics = {}
    
    # RMSE
    metrics['RMSE'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # R2
    try:
        metrics['R2'] = float(r2_score(y_true, y_pred))
    except:
        metrics['R2'] = np.nan
    
    # MAE
    try:
        metrics['MAE'] = float(mean_absolute_error(y_true, y_pred))
    except:
        metrics['MAE'] = np.nan
    
    # Pearson correlation
    try:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        metrics['Pearson_r'] = float(pearson_r)
        metrics['Pearson_p'] = float(pearson_p)
    except:
        metrics['Pearson_r'] = np.nan
        metrics['Pearson_p'] = np.nan
    
    # Spearman correlation
    try:
        spearman_rho, spearman_p = spearmanr(y_true, y_pred)
        metrics['Spearman_rho'] = float(spearman_rho)
        metrics['Spearman_p'] = float(spearman_p)
    except:
        metrics['Spearman_rho'] = np.nan
        metrics['Spearman_p'] = np.nan
    
    # Explained Variance
    try:
        metrics['Explained_Variance'] = float(explained_variance_score(y_true, y_pred))
    except:
        metrics['Explained_Variance'] = np.nan
    
    return metrics


def compute_extended_classification_metrics(y_true, y_pred_probs):
    """Compute extended classification metrics"""
    metrics = {}
    
    # Binary predictions (threshold at 0.5)
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)
    
    # AUC and PR-AUC
    try:
        metrics['AUC'] = float(roc_auc_score(y_true, y_pred_probs))
        metrics['PR_AUC'] = float(average_precision_score(y_true, y_pred_probs))
    except:
        metrics['AUC'] = np.nan
        metrics['PR_AUC'] = np.nan
    
    # Accuracy
    try:
        metrics['Accuracy'] = float(accuracy_score(y_true, y_pred_binary))
    except:
        metrics['Accuracy'] = np.nan
    
    # Precision
    try:
        metrics['Precision'] = float(precision_score(y_true, y_pred_binary, zero_division=0))
    except:
        metrics['Precision'] = np.nan
    
    # Recall
    try:
        metrics['Recall'] = float(recall_score(y_true, y_pred_binary, zero_division=0))
    except:
        metrics['Recall'] = np.nan
    
    # F1
    try:
        metrics['F1'] = float(f1_score(y_true, y_pred_binary, zero_division=0))
    except:
        metrics['F1'] = np.nan
    
    # Specificity
    try:
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        metrics['Specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    except:
        metrics['Specificity'] = np.nan
    
    return metrics


def predict_with_chemprop(model_path, test_data_path, features_path=None):
    """
    Use Chemprop's high-level API to make predictions
    This ensures consistency with training-time predictions
    
    Args:
        model_path: Path to trained Chemprop model checkpoint
        test_data_path: Path to test CSV file
        features_path: Optional path to features file
    
    Returns:
        predictions: numpy array of predictions
    """
    import tempfile
    import os
    from chemprop.train import make_predictions
    from chemprop.args import PredictArgs
    
    print(f"  Loading Chemprop model from: {model_path}")
    print(f"  Loading test data from: {test_data_path}")
    
    # Create temporary output file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        preds_path = f.name
    
    try:
        # Prepare prediction arguments
        args_list = [
            '--test_path', str(test_data_path),
            '--checkpoint_paths', str(model_path),
            '--preds_path', preds_path,
            '--num_workers', '0'  # ← 添加这行：禁用多进程以避免共享内存问题
        ]
        
        # Add features path if provided
        if features_path and Path(features_path).exists():
            args_list.extend(['--features_path', str(features_path)])
            print(f"  Using features from: {features_path}")
        
        # Parse arguments
        args = PredictArgs().parse_args(args_list)
        
        # Make predictions using Chemprop's high-level API
        print(f"  Making predictions...")
        preds = make_predictions(args)
        
        # Convert to numpy array
        preds_array = np.array(preds)
        print(f"  Predictions shape: {preds_array.shape}")
        
        return preds_array
        
    finally:
        # Clean up temporary file
        if os.path.exists(preds_path):
            os.remove(preds_path)


def evaluate_fold(fold_idx, split_dir, output_dir, task_types):
    """
    Evaluate a single fold using Chemprop models
    
    Args:
        fold_idx: Fold index (0-4)
        split_dir: Path to split directory
        output_dir: Path to output directory
        task_types: Dict mapping task names to 'regression' or 'classification'
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Fold {fold_idx}")
    print(f"{'='*80}\n")
    
    split_path = Path(split_dir) / f'cv_{fold_idx}'
    fold_output = Path(output_dir) / f'fold_{fold_idx}'
    fold_output.mkdir(parents=True, exist_ok=True)
    
    # Separate tasks
    reg_tasks = [t for t, typ in task_types.items() if typ == 'regression']
    clf_tasks = [t for t, typ in task_types.items() if typ == 'classification']
    
    # Load test data to get ground truth
    test_csv = split_path / 'test.csv'
    test_df = pd.read_csv(test_csv)
    
    reg_scores = {}
    clf_scores = {}
    
    # Evaluate regression tasks
    if reg_tasks:
        print("\n[Regression Tasks]")
        model_path = split_path / 'fold_0' / 'model_0' / 'model.pt'
        features_path = split_path / 'test_extra_x.csv'
        
        if model_path.exists():
            # Make predictions
            preds = predict_with_chemprop(
                str(model_path),
                str(test_csv),
                str(features_path) if features_path.exists() else None
            )
            
            # Compute metrics for each task
            for i, task in enumerate(reg_tasks):
                if task not in test_df.columns:
                    continue
                
                # Get valid data
                mask = test_df[task].notna()
                if mask.sum() < 2:
                    continue
                
                y_true = test_df.loc[mask, task].values
                y_pred = preds[mask, i]
                
                metrics = compute_extended_regression_metrics(y_true, y_pred)
                reg_scores[task] = metrics
                print(f"    {task}: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
        else:
            print(f"  Warning: Model not found at {model_path}")
    
    # Evaluate classification tasks
    if clf_tasks:
        print("\n[Classification Tasks]")
        # Use separate classification model from cv_X_clf directory
        clf_split_dir = Path(split_dir) / f'cv_{fold_idx}_clf'
        clf_model_path = clf_split_dir / 'fold_0' / 'model_0' / 'model.pt'
        # Test data is in the original cv_X directory
        test_clf_csv = split_path / 'test_clf.csv' if (split_path / 'test_clf.csv').exists() else test_csv
        features_clf_path = split_path / 'test_clf_extra_x.csv'
        
        if clf_model_path.exists():
            # Load test data for classification
            test_clf_df = pd.read_csv(test_clf_csv)
            
            # Make predictions
            preds_logits = predict_with_chemprop(
                str(clf_model_path),
                str(test_clf_csv),
                str(features_clf_path) if features_clf_path.exists() else None
            )
            
            # Convert logits to probabilities
            preds_probs = sigmoid(preds_logits)
            
            print(f"  Predictions shape: {preds_probs.shape}")
            print(f"  Number of classification tasks in data: {len(clf_tasks)}")
            print(f"  Number of model outputs: {preds_probs.shape[1]}")
            
            # Process all available classification tasks (model should predict all 15 tasks)
            num_model_outputs = preds_probs.shape[1]
            print(f"  Expected classification tasks: {len(clf_tasks)}")
            print(f"  Model outputs: {num_model_outputs}")

            # Match tasks with model outputs
            tasks_to_process = []
            for task in clf_tasks:
                if task in test_clf_df.columns:
                    tasks_to_process.append(task)

            if len(tasks_to_process) != num_model_outputs:
                print(f"  Warning: Number of tasks ({len(tasks_to_process)}) does not match model outputs ({num_model_outputs})")
                print(f"  Will process {min(len(tasks_to_process), num_model_outputs)} tasks")
                tasks_to_process = tasks_to_process[:num_model_outputs]
            
            # Compute metrics for each task (using correct index)
            for i, task in enumerate(tasks_to_process):
                if i >= num_model_outputs:
                    break
                    
                # Get valid data
                mask = test_clf_df[task].notna()
                if mask.sum() < 2:
                    continue
                
                y_true = test_clf_df.loc[mask, task].values.astype(int)
                y_pred_probs = preds_probs[mask, i]  # Use index i which matches model output
                
                # Check if we have both classes
                if len(np.unique(y_true)) < 2:
                    continue
                
                metrics = compute_extended_classification_metrics(y_true, y_pred_probs)
                clf_scores[task] = metrics
                print(f"    {task}: AUC={metrics['AUC']:.4f}, Acc={metrics['Accuracy']:.4f}")
        else:
            print(f"  Warning: Classification model not found at {clf_model_path}")
            print(f"  Expected path: {clf_split_dir / 'fold_0' / 'model_0' / 'model.pt'}")
            print(f"  Make sure classification models were trained in cv_X_clf directories")
        
    # Save scores
    if reg_scores:
        reg_df = pd.DataFrame(reg_scores).T
        reg_df.index.name = 'Task'
        reg_file = fold_output / 'test_scores.csv'
        reg_df.to_csv(reg_file)
        print(f"\n  ✓ Regression scores saved to: {reg_file}")
    
    if clf_scores:
        clf_df = pd.DataFrame(clf_scores).T
        clf_df.index.name = 'Task'
        clf_file = fold_output / 'test_scores_clf.csv'
        clf_df.to_csv(clf_file)
        print(f"  ✓ Classification scores saved to: {clf_file}")
    
    return reg_scores, clf_scores


def aggregate_scores(output_dir, num_folds, task_types):
    """Aggregate scores across folds"""
    print(f"\n{'='*80}")
    print("Aggregating Cross-Validation Scores")
    print(f"{'='*80}\n")
    
    output_dir = Path(output_dir)
    
    # Separate tasks
    reg_tasks = [t for t, typ in task_types.items() if typ == 'regression']
    clf_tasks = [t for t, typ in task_types.items() if typ == 'classification']
    
    # Aggregate regression scores
    if reg_tasks:
        reg_scores_per_fold = []
        for fold in range(num_folds):
            fold_file = output_dir / f'fold_{fold}' / 'test_scores.csv'
            if fold_file.exists():
                df = pd.read_csv(fold_file, index_col=0)
                reg_scores_per_fold.append(df)
        
        if reg_scores_per_fold:
            all_scores = pd.concat(reg_scores_per_fold, axis=0)
            grouped = all_scores.groupby(all_scores.index)
            
            # Get all columns dynamically
            metric_cols = all_scores.columns
            agg_dict = {}
            for col in metric_cols:
                agg_dict[f'{col}_mean'] = grouped[col].mean()
                agg_dict[f'{col}_std'] = grouped[col].std()
            
            agg_df = pd.DataFrame(agg_dict)
            agg_file = output_dir / 'test_scores_reg_agg.csv'
            agg_df.to_csv(agg_file)
            print(f"  ✓ Aggregated regression scores saved to: {agg_file}")
    
    # Aggregate classification scores
    if clf_tasks:
        clf_scores_per_fold = []
        for fold in range(num_folds):
            fold_file = output_dir / f'fold_{fold}' / 'test_scores_clf.csv'
            if fold_file.exists():
                df = pd.read_csv(fold_file, index_col=0)
                clf_scores_per_fold.append(df)
        
        if clf_scores_per_fold:
            all_scores = pd.concat(clf_scores_per_fold, axis=0)
            grouped = all_scores.groupby(all_scores.index)
            
            # Get all columns dynamically
            metric_cols = all_scores.columns
            agg_dict = {}
            for col in metric_cols:
                agg_dict[f'{col}_mean'] = grouped[col].mean()
                agg_dict[f'{col}_std'] = grouped[col].std()
            
            agg_df = pd.DataFrame(agg_dict)
            agg_file = output_dir / 'test_scores_clf_agg.csv'
            agg_df.to_csv(agg_file)
            print(f"  ✓ Aggregated classification scores saved to: {agg_file}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate original Chemprop models with extended metrics'
    )
    parser.add_argument('--split_name', type=str, default='by_source_smiles',
                       help='Name of the split')
    parser.add_argument('--num_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    # Set paths
    split_dir = f'../data/crossval_splits/{args.split_name}'
    output_dir = f'../results/chemprop_model_extended/{args.split_name}'
    
    # Load target roles
    with open('../data/args_files/target_roles.json', 'r') as f:
        target_roles = json.load(f)
    
    reg_tasks = target_roles.get('regression', target_roles.get('regression_targets', []))
    clf_tasks = target_roles.get('classification', target_roles.get('classification_targets', []))
    
    task_types = {}
    for task in reg_tasks:
        task_types[task] = 'regression'
    for task in clf_tasks:
        task_types[task] = 'classification'
    
    print(f"\n{'='*80}")
    print("Chemprop Model Evaluation with Extended Metrics")
    print(f"{'='*80}")
    print(f"Split: {split_dir}")
    print(f"Output: {output_dir}")
    print(f"Folds: {args.num_folds}")
    print(f"Regression tasks: {len(reg_tasks)}")
    print(f"Classification tasks: {len(clf_tasks)}")
    print(f"{'='*80}")
    
    # Evaluate each fold
    for fold_idx in range(args.num_folds):
        evaluate_fold(fold_idx, split_dir, output_dir, task_types)
    
    # Aggregate scores
    aggregate_scores(output_dir, args.num_folds, task_types)
    
    print(f"\n{'='*80}")
    print("Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"  - test_scores_reg_agg.csv")
    print(f"  - test_scores_clf_agg.csv")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()