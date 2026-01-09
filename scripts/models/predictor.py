import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import json
from sklearn.metrics import (
    roc_auc_score, mean_squared_error, r2_score, average_precision_score, 
    roc_curve, mean_absolute_error, accuracy_score, precision_score, 
    recall_score, f1_score, explained_variance_score
)
from scipy.stats import pearsonr, spearmanr


class LNPPredictor:
    # Predictor for LNPMix model with score output compatible with original pipeline
    
    def __init__(self, 
                 model: nn.Module,
                 config,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device
        
        print(f"[Predictor] Initialized on {device}")
    
    def predict(self, data_loader: DataLoader) -> Tuple[Dict, Dict, Dict]:
        # Make predictions on dataset
        # Returns: (predictions, targets, masks)
        
        all_preds = {task: [] for task in self.config.task_names}
        all_targets = {task: [] for task in self.config.task_names}
        all_masks = {task: [] for task in self.config.task_names}
        all_smiles = []
        
        print("Making predictions...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                # Move to device
                tabular_data = {k: v.to(self.device) for k, v in batch['tabular_data'].items()}
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
                
                # Forward pass
                predictions = self.model(batch['smiles'], tabular_data)
                
                # Collect results
                all_smiles.extend(batch['smiles'])
                for task in self.config.task_names:
                    all_preds[task].append(predictions[task].cpu().numpy())
                    all_targets[task].append(targets[task].cpu().numpy())
                    all_masks[task].append(masks[task].cpu().numpy())
        
        # Concatenate all batches
        for task in self.config.task_names:
            all_preds[task] = np.concatenate(all_preds[task], axis=0)
            all_targets[task] = np.concatenate(all_targets[task], axis=0)
            all_masks[task] = np.concatenate(all_masks[task], axis=0)
        
        return all_preds, all_targets, all_masks, all_smiles
    
    def compute_scores(self, 
                      predictions: Dict,
                      targets: Dict,
                      masks: Dict,
                      output_dir: str,
                      fold_name: str = None):
        # Compute scores and save to CSV files
        # Compatible with original pipeline output format
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate regression and classification tasks
        reg_tasks = [t for t in self.config.task_names if self.config.task_types[t] == 'regression']
        clf_tasks = [t for t in self.config.task_names if self.config.task_types[t] == 'classification']
        
        # Compute regression scores
        if reg_tasks:
            reg_scores = self._compute_regression_scores(predictions, targets, masks, reg_tasks)
            reg_df = pd.DataFrame(reg_scores).T
            reg_df.index.name = 'Task'
            reg_file = output_dir / 'test_scores.csv'
            reg_df.to_csv(reg_file)
            print(f"Regression scores saved to {reg_file}")
        
        # Compute classification scores
        if clf_tasks:
            clf_scores = self._compute_classification_scores(predictions, targets, masks, clf_tasks)
            clf_df = pd.DataFrame(clf_scores).T
            clf_df.index.name = 'Task'
            clf_file = output_dir / 'test_scores_clf.csv'
            clf_df.to_csv(clf_file)
            print(f"Classification scores saved to {clf_file}")
        
        return reg_scores if reg_tasks else {}, clf_scores if clf_tasks else {}
    
    def _compute_regression_scores(self, predictions, targets, masks, task_names):
        # Compute comprehensive regression metrics
        scores = {}
        
        for task in task_names:
            mask = masks[task].flatten() == 1
            if mask.sum() == 0:
                scores[task] = {
                    'RMSE': np.nan, 'R2': np.nan, 'MAE': np.nan,
                    'Pearson_r': np.nan, 'Pearson_p': np.nan,
                    'Spearman_rho': np.nan, 'Spearman_p': np.nan,
                    'Explained_Variance': np.nan
                }
                continue
            
            pred = predictions[task].flatten()[mask]
            target = targets[task].flatten()[mask]
            
            # RMSE
            rmse = float(np.sqrt(np.mean((target - pred)**2)))
            
            # R2
            try:
                r2 = r2_score(target, pred)
            except:
                r2 = np.nan
            
            # MAE
            try:
                mae = mean_absolute_error(target, pred)
            except:
                mae = np.nan
            
            # Pearson correlation
            try:
                pearson_r, pearson_p = pearsonr(target, pred)
            except:
                pearson_r, pearson_p = np.nan, np.nan
            
            # Spearman correlation
            try:
                spearman_rho, spearman_p = spearmanr(target, pred)
            except:
                spearman_rho, spearman_p = np.nan, np.nan
        
            # Explained Variance
            try:
                explained_var = explained_variance_score(target, pred)
            except:
                explained_var = np.nan
            
            scores[task] = {
                'RMSE': rmse,
                'R2': r2,
                'MAE': mae,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_rho': spearman_rho,
                'Spearman_p': spearman_p,
                'Explained_Variance': explained_var
            }
        
        return scores
    
    def _compute_classification_scores(self, predictions, targets, masks, task_names):
        # Compute comprehensive classification metrics
        scores = {}
        
        for task in task_names:
            mask = masks[task].flatten() == 1
            if mask.sum() == 0:
                scores[task] = {
                    'AUC': np.nan, 'PR_AUC': np.nan, 'Accuracy': np.nan,
                    'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan,
                    'Specificity': np.nan
                }
                continue
        
            pred_probs = predictions[task].flatten()[mask]
            target = targets[task].flatten()[mask]
        
            # Binary predictions (threshold at 0.5)
            pred_binary = (pred_probs >= 0.5).astype(int)
            
            # Initialize metrics
            auc = pr_auc = accuracy = precision = recall = f1 = specificity = np.nan
            
            # Check if we have at least 2 classes
            if len(np.unique(target)) >= 2:
                try:
                    auc = float(roc_auc_score(target, pred_probs))
                    pr_auc = float(average_precision_score(target, pred_probs))
                except Exception as e:
                    pass
                
                try:
                    accuracy = float(accuracy_score(target, pred_binary))
                except:
                    pass
                
                try:
                    precision = float(precision_score(target, pred_binary, zero_division=0))
                except:
                    pass
                
                try:
                    recall = float(recall_score(target, pred_binary, zero_division=0))
                except:
                    pass
                
                try:
                    f1 = float(f1_score(target, pred_binary, zero_division=0))
                except:
                    pass
                
                try:
                    # Specificity = TN / (TN + FP)
                    tn = np.sum((target == 0) & (pred_binary == 0))
                    fp = np.sum((target == 0) & (pred_binary == 1))
                    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
                except:
                    pass
            
            scores[task] = {
                'AUC': auc,
                'PR_AUC': pr_auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Specificity': specificity
            }
        
        return scores
    
    def save_predictions(self,
                        predictions: Dict,
                        targets: Dict,
                        masks: Dict,
                        smiles: List[str],
                        output_file: str,
                        apply_biodist_transform: bool = True):
        """
        Save predictions to CSV file with optional Biodistribution->Delivery_target transform
        
        Args:
            predictions: Dictionary of predictions for each task
            targets: Dictionary of target values
            masks: Dictionary of valid sample masks
            smiles: List of SMILES strings
            output_file: Path to output CSV file
            apply_biodist_transform: Whether to apply unified post-processing
        """
        
        data = {'smiles': smiles}
        
        # Save raw predictions and targets
        for task in self.config.task_names:
            pred = predictions[task].flatten()
            target = targets[task].flatten()
            mask = masks[task].flatten()
            
            data[f'{task}_pred'] = pred
            data[f'{task}_true'] = target
            data[f'{task}_mask'] = mask
        
        df = pd.DataFrame(data)
        
        # ==================== UNIFIED POST-PROCESSING ====================
        # Apply same post-processing as Chemprop model for fair comparison
        # =================================================================
        
        if apply_biodist_transform:
            from biodistribution_to_target import apply_postprocessing_to_predictions
            from postprocessing_config import get_organ_thresholds
            
            # Use unified threshold configuration (same as Chemprop)
            organ_thresholds = get_organ_thresholds('default')
            
            # Apply post-processing
            # Note: Attention model predictions don't have 'cv_X_pred_' prefix
            df = apply_postprocessing_to_predictions(
                df,
                organ_thresholds=organ_thresholds,
                prediction_prefix='',  # No prefix for attention model
                inplace=True
            )
        
        # ====================== END POST-PROCESSING ======================
        
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        if apply_biodist_transform:
            n_biodist = len([c for c in df.columns if 'Biodistribution' in c and '_pred' in c])
            n_target = len([c for c in df.columns if 'Delivery_target' in c and '_pred' in c])
            print(f"  Post-processing: {n_biodist} Biodistribution -> {n_target} Delivery_target columns")


def aggregate_cv_scores(cv_dir: str, num_folds: int, task_types: Dict[str, str]):
    # Aggregate cross-validation scores across folds
    
    cv_dir = Path(cv_dir)
    
    # Separate regression and classification tasks
    reg_tasks = [t for t, typ in task_types.items() if typ == 'regression']
    clf_tasks = [t for t, typ in task_types.items() if typ == 'classification']
    
    # Aggregate regression scores
    if reg_tasks:
        reg_scores_per_fold = []
        for fold in range(num_folds):
            fold_file = cv_dir / f'fold_{fold}' / 'test_scores.csv'
            if fold_file.exists():
                df = pd.read_csv(fold_file, index_col=0)
                reg_scores_per_fold.append(df)
        
        if reg_scores_per_fold:
            # Compute mean and std for all metrics
            all_scores = pd.concat(reg_scores_per_fold, axis=0)
            grouped = all_scores.groupby(all_scores.index)
            
            # Get all columns dynamically
            metric_cols = all_scores.columns
            agg_dict = {}
            for col in metric_cols:
                agg_dict[f'{col}_mean'] = grouped[col].mean()
                agg_dict[f'{col}_std'] = grouped[col].std()
            
            agg_df = pd.DataFrame(agg_dict)
            
            agg_file = cv_dir / 'test_scores_reg_agg.csv'
            agg_df.to_csv(agg_file)
            print(f"Aggregated regression scores saved to {agg_file}")
    
    # Aggregate classification scores
    if clf_tasks:
        clf_scores_per_fold = []
        for fold in range(num_folds):
            fold_file = cv_dir / f'fold_{fold}' / 'test_scores_clf.csv'
            if fold_file.exists():
                df = pd.read_csv(fold_file, index_col=0)
                clf_scores_per_fold.append(df)
        
        if clf_scores_per_fold:
            # Compute mean and std for all metrics
            all_scores = pd.concat(clf_scores_per_fold, axis=0)
            grouped = all_scores.groupby(all_scores.index)
            
            # Get all columns dynamically
            metric_cols = all_scores.columns
            agg_dict = {}
            for col in metric_cols:
                agg_dict[f'{col}_mean'] = grouped[col].mean()
                agg_dict[f'{col}_std'] = grouped[col].std()
            
            agg_df = pd.DataFrame(agg_dict)
            
            agg_file = cv_dir / 'test_scores_clf_agg.csv'
            agg_df.to_csv(agg_file)
            print(f"Aggregated classification scores saved to {agg_file}")