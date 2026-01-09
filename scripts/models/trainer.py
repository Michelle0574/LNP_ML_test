import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, average_precision_score
import time


class LNPDataset(Dataset):
    # Dataset for LNP training
    
    def __init__(self, 
                 data_df: pd.DataFrame,
                 smiles_col: str,
                 task_names: List[str],
                 task_types: Dict[str, str],
                 feature_cols: Dict[str, List[str]]):
        self.data_df = data_df.reset_index(drop=True)
        self.smiles_col = smiles_col
        self.task_names = task_names
        self.task_types = task_types
        self.feature_cols = feature_cols
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        
        # SMILES
        smiles = row[self.smiles_col]
        
        # Tabular features
        tabular_data = {}
        for group_name, cols in self.feature_cols.items():
            features = []
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    val = 0.0  # Handle missing values
                features.append(float(val))
            tabular_data[group_name] = torch.tensor(features, dtype=torch.float32)
        
        # Targets and masks
        targets = {}
        masks = {}
        for task_name in self.task_names:
            if task_name in row and not pd.isna(row[task_name]):
                targets[task_name] = torch.tensor([float(row[task_name])], dtype=torch.float32)
                masks[task_name] = torch.tensor([1.0], dtype=torch.float32)
            else:
                targets[task_name] = torch.tensor([0.0], dtype=torch.float32)
                masks[task_name] = torch.tensor([0.0], dtype=torch.float32)
        
        return {
            'smiles': smiles,
            'tabular_data': tabular_data,
            'targets': targets,
            'masks': masks
        }


def collate_fn(batch):
    # Custom collate function
    smiles_list = [item['smiles'] for item in batch]
    
    # Stack tabular features
    tabular_data = {}
    for key in batch[0]['tabular_data'].keys():
        tabular_data[key] = torch.stack([item['tabular_data'][key] for item in batch])
    
    # REMOVE batch-wise normalization (causes issues with varying batch sizes)
    # Features will be normalized by BatchNorm1d in TokenProjector
    for key in tabular_data.keys():
        feat = tabular_data[key]
        
        # Step 1: Replace NaN and Inf
        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Step 2: Clip extreme values
        feat = torch.clamp(feat, min=-100.0, max=100.0)
        
        # REMOVED: Z-score normalization here
        # Let TokenProjector's BatchNorm handle it
        
        tabular_data[key] = feat
    
    # Stack targets and masks
    targets = {}
    masks = {}
    for task_name in batch[0]['targets'].keys():
        targets[task_name] = torch.stack([item['targets'][task_name] for item in batch])
        masks[task_name] = torch.stack([item['masks'][task_name] for item in batch])
    
    return {
        'smiles': smiles_list,
        'tabular_data': tabular_data,
        'targets': targets,
        'masks': masks
    }


class LNPTrainer:
    # Trainer for LNPMix model with Chemprop-style logging
    
    def __init__(self, 
                 model: nn.Module,
                 config,
                 save_dir: str,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Track token weights if enabled
        self.track_token_weights = getattr(config, 'use_token_weights', False)
        if self.track_token_weights:
            self.token_weight_history = []
            print(f"[Trainer] Token weight tracking enabled")
        
        # Gradient clipping configuration
        self.gradient_clip_norm = getattr(config, 'gradient_clip_norm', 1.0)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        print(f"\n[Trainer] Initialized")
        print(f"  Device: {device}")
        print(f"  Save dir: {save_dir}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Batch size: {config.batch_size}")
    
    def _compute_task_weights(self, train_loader):
        """
        Compute task-specific loss weights based on:
        1. Sample count (fewer samples = higher weight)
        2. Task difficulty (classification gets higher weight)
        """
        task_counts = {}
        
        # Count samples per task
        for batch in train_loader:
            for task_name in batch['targets'].keys():
                mask = batch['masks'][task_name]
                task_counts[task_name] = task_counts.get(task_name, 0) + mask.sum().item()
        
        # Compute weights: inverse frequency, normalized
        max_count = max(task_counts.values())
        task_weights = {}
        
        for task_name, count in task_counts.items():
            # Base weight: inverse frequency
            base_weight = max_count / count
            
            # Boost classification tasks (they need more attention)
            if self.config.task_types[task_name] == 'classification':
                base_weight *= 1.5  # 50% boost for classification
            
            task_weights[task_name] = base_weight
        
        # Normalize so average weight is 1.0
        avg_weight = sum(task_weights.values()) / len(task_weights)
        task_weights = {k: v/avg_weight for k, v in task_weights.items()}
        
        return task_weights

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        print(f"\n[Trainer] Initialized")
        print(f"  Device: {device}")
        print(f"  Save dir: {save_dir}")
        print(f"  Learning rate: {config.learning_rate}")
        print(f"  Batch size: {config.batch_size}")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        # Train for one epoch
        self.model.train()
        
        total_loss = 0.0
        task_losses = {task: [] for task in self.config.task_names}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            tabular_data = {k: v.to(self.device) for k, v in batch['tabular_data'].items()}
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
            
            # Debug: Check for NaN in first batch
            if batch_idx == 0 and epoch == 1:
                print(f"\n[Debug] First batch check:")
                for k, v in tabular_data.items():
                    if torch.isnan(v).any():
                        print(f"  WARNING: NaN in tabular_data[{k}]")
                    print(f"  {k}: shape={v.shape}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
                for k, v in targets.items():
                    valid_mask = masks[k] == 1
                    if valid_mask.sum() > 0:
                        valid_targets = v[valid_mask]
                        print(f"  target[{k}]: {valid_mask.sum().item()} valid samples, range=[{valid_targets.min().item():.4f}, {valid_targets.max().item():.4f}]")
            
            # Forward pass
            predictions = self.model(batch['smiles'], tabular_data)
            
            # Check predictions immediately after forward
            if batch_idx == 0 and epoch == 1:
                print(f"\n[Debug] First forward pass complete")
                for task_name, pred in predictions.items():
                    has_nan = torch.isnan(pred).any().item()
                    if has_nan:
                        print(f"  {task_name}: HAS NaN!")
                    else:
                        print(f"  {task_name}: OK, range=[{pred.min().item():.4f}, {pred.max().item():.4f}]")
                print()
            
            # Compute loss
            loss, loss_dict = self.model.get_loss(predictions, targets, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.gradient_clip_norm
            )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            for task, task_loss in loss_dict.items():
                if task_loss > 0:
                    task_losses[task].append(task_loss)
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute average losses
        avg_loss = total_loss / len(train_loader)
        avg_task_losses = {task: np.mean(losses) if losses else 0.0 
                          for task, losses in task_losses.items()}
        
        return avg_loss, avg_task_losses
    
    def evaluate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict, Dict]:
        # Evaluate on validation set
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {task: [] for task in self.config.task_names}
        
        # Collect predictions and targets for metrics
        all_preds = {task: [] for task in self.config.task_names}
        all_targets = {task: [] for task in self.config.task_names}
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Valid]")
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                tabular_data = {k: v.to(self.device) for k, v in batch['tabular_data'].items()}
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
                
                # Forward pass
                predictions = self.model(batch['smiles'], tabular_data)
                
                # Compute loss
                loss, loss_dict = self.model.get_loss(predictions, targets, masks)
                
                total_loss += loss.item()
                for task, task_loss in loss_dict.items():
                    if task_loss > 0:
                        task_losses[task].append(task_loss)
                
                # Collect predictions and targets
                for task in self.config.task_names:
                    mask = masks[task].cpu().numpy()
                    if mask.sum() > 0:
                        pred = predictions[task].cpu()
                        
                        # Apply sigmoid for classification tasks
                        if self.config.task_types[task] == 'classification':
                            pred = torch.sigmoid(pred)
                        
                        pred = pred.numpy()[mask == 1]
                        target = targets[task].cpu().numpy()[mask == 1]
                        all_preds[task].extend(pred.flatten())
                        all_targets[task].extend(target.flatten())
                
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Compute average losses
        avg_loss = total_loss / len(val_loader)
        avg_task_losses = {task: np.mean(losses) if losses else 0.0 
                          for task, losses in task_losses.items()}
        
        # Compute metrics
        metrics = self.compute_metrics(all_preds, all_targets)
        
        return avg_loss, avg_task_losses, metrics
    
    def compute_metrics(self, predictions: Dict[str, List], targets: Dict[str, List]) -> Dict[str, Dict]:
        # Compute evaluation metrics for each task
        metrics = {}
        
        for task in self.config.task_names:
            if len(predictions[task]) == 0:
                continue
            
            pred = np.array(predictions[task])
            target = np.array(targets[task])
            
            # Skip if predictions contain NaN
            if np.isnan(pred).any() or np.isnan(target).any():
                print(f"[WARNING] Skipping metrics for task {task} due to NaN values")
                if self.config.task_types[task] == 'regression':
                    metrics[task] = {'RMSE': float('nan'), 'R2': float('nan')}
                else:
                    metrics[task] = {'AUC': float('nan'), 'PR_AUC': float('nan')}
                continue
            
            task_type = self.config.task_types[task]
            
            if task_type == 'regression':
                rmse = np.sqrt(mean_squared_error(target, pred))
                r2 = r2_score(target, pred)
                metrics[task] = {'RMSE': rmse, 'R2': r2}
            else:  # classification
                try:
                    auc = roc_auc_score(target, pred)
                    pr_auc = average_precision_score(target, pred)
                    metrics[task] = {'AUC': auc, 'PR_AUC': pr_auc}
                except:
                    metrics[task] = {'AUC': 0.0, 'PR_AUC': 0.0}
        
        return metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int):
        # Main training loop
        
        # Compute task-adaptive loss weights if enabled
        if getattr(self.config, 'use_task_loss_weights', False):
            self.task_loss_weights = self._compute_task_weights(train_loader)
            print("\n[Trainer] Task Loss Weights:")
            for task, weight in sorted(self.task_loss_weights.items())[:10]:
                print(f"  {task}: {weight:.4f}")
        else:
            self.task_loss_weights = {task: 1.0 for task in self.config.task_names}
        
        print(f"\n{'='*80}")
        print(f"Starting Training for {epochs} epochs")
        print(f"{'='*80}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_task_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_task_losses, val_metrics = self.evaluate(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metrics'].append(val_metrics)
            
            # Print epoch summary (Chemprop-style)
            epoch_time = time.time() - epoch_start_time
            self.print_epoch_summary(epoch, epochs, train_loss, val_loss, 
                                    train_task_losses, val_task_losses, 
                                    val_metrics, epoch_time)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                print(f"  → Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Log token weights periodically
            if self.track_token_weights and hasattr(self.model, 'get_token_importance'):
                weights = self.model.get_token_importance()
                if weights:
                    self.token_weight_history.append({
                        'epoch': epoch,
                        'weights': weights
                    })
                    
                    # Print weights every 5 epochs
                    if epoch % 5 == 0:
                        print(f"\n[Token Weights] Epoch {epoch}:")
                        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                        for name, weight in sorted_weights:
                            print(f"  {name:15s}: {weight:.4f}")
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            # Save checkpoint every N epochs
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\n{'='*80}")
        print(f"Training Completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*80}\n")
        
        # Save training history
        self.save_history()
    
    def print_epoch_summary(self, epoch, total_epochs, train_loss, val_loss,
                           train_task_losses, val_task_losses, val_metrics, epoch_time):
        # Print epoch summary in Chemprop style
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{total_epochs} | Time: {epoch_time:.2f}s")
        print(f"{'-'*80}")
        
        # Overall losses
        print(f"Overall Loss:")
        print(f"  Train: {train_loss:.4f} | Valid: {val_loss:.4f}")
        
        # Task-specific losses
        print(f"\nTask-specific Losses:")
        for task in self.config.task_names:
            train_task_loss = train_task_losses.get(task, 0.0)
            val_task_loss = val_task_losses.get(task, 0.0)
            print(f"  {task:20s}: Train {train_task_loss:.4f} | Valid {val_task_loss:.4f}")
        
        # Validation metrics
        if val_metrics:
            print(f"\nValidation Metrics:")
            for task, metrics in val_metrics.items():
                task_type = self.config.task_types[task]
                if task_type == 'regression':
                    print(f"  {task:20s}: RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
                else:
                    print(f"  {task:20s}: AUC={metrics['AUC']:.4f}, PR_AUC={metrics['PR_AUC']:.4f}")
        
        print(f"{'='*80}")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config.__dict__
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pt'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
    

    def save_history(self):
        # Save training history
        history_path = self.save_dir / 'training_history.json'
        
        # Convert all numpy/torch types to Python types for JSON serialization
        def convert_to_python(obj):
            if isinstance(obj, dict):
                return {k: convert_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(x) for x in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # torch tensor scalar
                return obj.item()
            else:
                return obj
        
        history_json = {
            'train_loss': [float(x) for x in self.history['train_loss']],
            'val_loss': [float(x) for x in self.history['val_loss']],
            'val_metrics': convert_to_python(self.history['val_metrics'])
        }
        
        # Add token weight history if tracking
        if self.track_token_weights and hasattr(self, 'token_weight_history') and self.token_weight_history:
            history_json['token_weights'] = self.token_weight_history

        with open(history_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        print(f"Training history saved to {history_path}")

    
    def load_checkpoint(self, checkpoint_path: str):
        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']