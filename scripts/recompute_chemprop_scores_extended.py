#!/usr/bin/env python3
"""
Recompute and extend original Chemprop model scores with additional metrics
Reads original predictions and computes extended metrics
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, explained_variance_score
)
from scipy.stats import pearsonr, spearmanr


def sigmoid(x):
    """Sigmoid function to convert logits to probabilities"""
    return 1 / (1 + np.exp(-x))


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


def compute_extended_classification_metrics(y_true, y_pred_logits):
    """Compute extended classification metrics from logits"""
    metrics = {}
    
    # Convert logits to probabilities using sigmoid
    y_pred_probs = sigmoid(y_pred_logits)
    
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


def process_split(split_name, num_folds=5):
    """Process a cross-validation split and compute extended metrics"""
    
    base_dir = Path(f'../results/crossval_splits/{split_name}')
    output_dir = base_dir / 'crossval_performance_extended'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nProcessing split: {split_name}")
    print(f"Output directory: {output_dir}")
    
    # Load target roles to identify task types
    import json
    with open('../data/args_files/target_roles.json', 'r') as f:
        target_roles = json.load(f)
    
    reg_tasks = target_roles.get('regression', target_roles.get('regression_targets', []))
    clf_tasks = target_roles.get('classification', target_roles.get('classification_targets', []))
    
    print(f"\nRegression tasks: {len(reg_tasks)}")
    print(f"Classification tasks: {len(clf_tasks)}")
    
    # Process regression tasks
    if reg_tasks:
        reg_metrics_per_fold = {task: [] for task in reg_tasks}
        
        for fold in range(num_folds):
            fold_file = base_dir / f'cv_{fold}' / 'predicted_vs_actual.csv'
            
            if not fold_file.exists():
                print(f"  Warning: {fold_file} not found, skipping fold {fold}")
                continue
            
            print(f"  Processing fold {fold}...")
            df = pd.read_csv(fold_file)
            
            for task in reg_tasks:
                if task not in df.columns:
                    continue
                
                pred_col = f'cv_{fold}_pred_{task}'
                if pred_col not in df.columns:
                    continue
                
                # Get valid data (non-NaN)
                mask = df[task].notna() & df[pred_col].notna()
                if mask.sum() < 2:
                    continue
                
                y_true = df.loc[mask, task].values
                y_pred = df.loc[mask, pred_col].values
                
                metrics = compute_extended_regression_metrics(y_true, y_pred)
                reg_metrics_per_fold[task].append(metrics)
        
        # Aggregate regression metrics
        reg_agg_data = []
        for task in reg_tasks:
            if not reg_metrics_per_fold[task]:
                continue
            
            # Convert list of dicts to DataFrame
            task_df = pd.DataFrame(reg_metrics_per_fold[task])
            
            row = {'Task': task}
            for metric in task_df.columns:
                row[f'{metric}_mean'] = task_df[metric].mean()
                row[f'{metric}_std'] = task_df[metric].std()
            
            reg_agg_data.append(row)
        
        if reg_agg_data:
            reg_agg_df = pd.DataFrame(reg_agg_data)
            reg_agg_df.set_index('Task', inplace=True)
            reg_agg_file = output_dir / 'test_scores_reg_agg.csv'
            reg_agg_df.to_csv(reg_agg_file)
            print(f"\n  ✓ Regression scores saved to: {reg_agg_file}")
            print(f"    Tasks processed: {len(reg_agg_data)}")
    
    # Process classification tasks
    if clf_tasks:
        clf_metrics_per_fold = {task: [] for task in clf_tasks}
        
        for fold in range(num_folds):
            fold_file = base_dir / f'cv_{fold}' / 'predicted_vs_actual.csv'
            
            if not fold_file.exists():
                continue
            
            df = pd.read_csv(fold_file)
            
            # Get all prediction columns for this fold
            pred_cols = [col for col in df.columns if col.startswith(f'cv_{fold}_pred_')]
            
            for task in clf_tasks:
                if task not in df.columns:
                    continue
                
                # Find the FIRST occurrence of prediction column
                pred_col = f'cv_{fold}_pred_{task}'
                matching_cols = [col for col in pred_cols if col == pred_col]
                
                if not matching_cols:
                    continue
                
                # Use only first occurrence (pandas will use first column with duplicate name)
                # Get valid data
                mask = df[task].notna() & df[pred_col].notna()
                if mask.sum() < 2:
                    continue
                
                y_true = df.loc[mask, task].values.astype(int)
                y_pred_logits = df.loc[mask, pred_col].values
                
                # Check if we have both classes
                if len(np.unique(y_true)) < 2:
                    continue
                
                metrics = compute_extended_classification_metrics(y_true, y_pred_logits)
                clf_metrics_per_fold[task].append(metrics)
        
        # Aggregate classification metrics
        clf_agg_data = []
        for task in clf_tasks:
            if not clf_metrics_per_fold[task]:
                continue
            
            task_df = pd.DataFrame(clf_metrics_per_fold[task])
            
            row = {'Task': task}
            for metric in task_df.columns:
                row[f'{metric}_mean'] = task_df[metric].mean()
                row[f'{metric}_std'] = task_df[metric].std()
            
            clf_agg_data.append(row)
        
        if clf_agg_data:
            clf_agg_df = pd.DataFrame(clf_agg_data)
            clf_agg_df.set_index('Task', inplace=True)
            clf_agg_file = output_dir / 'test_scores_clf_agg.csv'
            clf_agg_df.to_csv(clf_agg_file)
            print(f"\n  ✓ Classification scores saved to: {clf_agg_file}")
            print(f"    Tasks processed: {len(clf_agg_data)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python recompute_chemprop_scores_extended.py <split_name>")
        print("Example: python recompute_chemprop_scores_extended.py by_source_smiles")
        sys.exit(1)
    
    split_name = sys.argv[1]
    process_split(split_name, num_folds=5)
    
    print("\n✓ Extended metrics computation complete!")