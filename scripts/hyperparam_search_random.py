import json
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import argparse
import copy

# Optimized search space based on V1 trial_18 success
# Key insight: V2 failed because weight_decay was too high (0.001-0.01)
# V1 trial_18 used: d_model=64, heads=4, dropout=0.26, wd=5.3e-5, lr=0.0002
search_space = {
    # Model architecture - explore around V1 success
    'd_model': [64, 128, 256],              # Keep 64 as it worked well
    'num_heads': [2, 4, 8],                 # V1 used 4
    
    # Regularization - CRITICAL: much lower weight_decay!
    'dropout': (0.15, 0.35),                # V1: 0.26 ✓
    'weight_decay': (1e-5, 5e-4),           # V1: 5.3e-5 ✓ (NOT 0.001-0.01!)
    
    # Cross-attention layers - V1 used 3
    'cross_attention_layers': [2, 3, 4],    # Explore around 3
    
    # Learning rate - V1 used 0.0002
    'learning_rate': (5e-5, 5e-4),          # V1 range worked well
    
    Separate dropouts for classification vs regression
    'classification_dropout': (0.1, 0.25),  # Lower for classification
    'regression_dropout': (0.2, 0.35),      # Can be higher
    
    Label smoothing for classification
    'label_smoothing': [0.0, 0.05],         # Avoid 0.1, too aggressive
    
    Early stopping patience
    'early_stopping_patience': [20, 25, 30],
    
    Token weight initialization strategy
    'token_weight_init_strategy': ['balanced', 'chem_focused'],
}

def sample_hyperparameters(search_space, seed):
    """Sample hyperparameters from search space"""
    np.random.seed(seed)
    config = {}
    
    for key, value in search_space.items():
        if isinstance(value, list):
            sampled = np.random.choice(value)
            if hasattr(sampled, 'item'):
                config[key] = sampled.item()
            else:
                config[key] = sampled
        elif isinstance(value, tuple):
            low, high = value
            config[key] = float(np.random.uniform(low, high))
    
    return config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Hyperparameter search for attention model')
parser.add_argument('--num_trials', type=int, default=20,
                   help='Number of trials to run')
parser.add_argument('--start_trial', type=int, default=0,
                   help='Starting trial number (for continuation)')
parser.add_argument('--output_dir', type=str, default='../results/hyperparam_search',
                   help='Output directory for search results')
parser.add_argument('--search_type', type=str, default='both', 
                   choices=['regression', 'classification', 'both'],
                   help='Type of search: regression, classification, or both')
args = parser.parse_args()

# Load base config
base_config_path = '../data/args_files/attention_config.json'
with open(base_config_path, 'r') as f:
    base_config = json.load(f)

# Create search results directory
search_results_dir = Path(args.output_dir)
search_results_dir.mkdir(parents=True, exist_ok=True)

n_trials = args.num_trials
start_trial = args.start_trial
search_type = args.search_type

print(f"\n{'='*80}")
print(f"Hyperparameter Search Configuration")
print(f"{'='*80}")
print(f"Search type: {search_type}")
print(f"Number of trials: {n_trials}")
print(f"Starting trial: {start_trial}")
print(f"Ending trial: {start_trial + n_trials - 1}")
print(f"Output directory: {search_results_dir}")
print(f"{'='*80}\n")

results = []

# Track best models
best_regression = {'rmse': float('inf'), 'trial': None, 'config': None}
best_classification = {'auc': 0.0, 'trial': None, 'config': None}

for i in range(n_trials):
    trial = start_trial + i
    
    # Sample hyperparameters
    hyper = sample_hyperparameters(search_space, seed=trial)
    
    # Update config with sampled hyperparameters
    config = copy.deepcopy(base_config)
    
    # Architecture
    config['d_model'] = int(hyper['d_model'])
    config['num_heads'] = int(hyper['num_heads'])
    config['cross_attention']['layers'] = int(hyper['cross_attention_layers'])
    
    # Ensure num_heads divides d_model
    while config['d_model'] % config['num_heads'] != 0:
        config['num_heads'] = int(np.random.choice([2, 4, 8]))
    
    # Regularization - CRITICAL: use lower weight_decay range
    config['dropout'] = float(hyper['dropout'])
    config['cross_attention']['dropout'] = float(hyper['dropout'])
    config['training']['learning_rate'] = float(hyper['learning_rate'])
    config['training']['weight_decay'] = float(hyper['weight_decay'])
    
    # Separate task-specific dropouts
    config['classification_dropout'] = float(hyper['classification_dropout'])
    config['regression_dropout'] = float(hyper['regression_dropout'])
    config['heads']['classification']['dropout'] = float(hyper['classification_dropout'])
    config['heads']['regression']['dropout'] = float(hyper['regression_dropout'])
    
    # Label smoothing
    config['heads']['classification']['label_smoothing'] = float(hyper['label_smoothing'])
    
    # Early stopping
    config['training']['early_stopping_patience'] = int(hyper['early_stopping_patience'])
    
    # Token weight initialization strategy
    config['token_weight_init_strategy'] = hyper['token_weight_init_strategy']
    
    # Task loss weighting (enable by default)
    config['use_task_loss_weights'] = True
    
    print(f"\n{'='*80}")
    print(f"Trial {trial} (iteration {i+1}/{n_trials})")
    for k, v in hyper.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print(f"{'='*80}")
    
    # Train ONCE for both regression and classification
    print(f"\n[Trial {trial}] Training multi-task model...")
    
    # Save config (one config per trial)
    config_path = search_results_dir / f'config_trial_{trial}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train model
    cmd = ['python', 'train_attention_model.py', 'by_source_smiles',
        '--config', str(config_path), '--epochs', '30', '--fold', '0']

    try:
        result_code = subprocess.run(cmd, check=False)
        if result_code.returncode != 0:
            print(f"  ⚠️ Training failed with return code {result_code.returncode}")
            print(f"  Skipping trial {trial}...")
    except Exception as e:
        print(f"  ⚠️ Training failed with exception: {e}")
        print(f"  Skipping trial {trial}...")
    
    # Get output files
    default_output = Path('../results/attention_model/by_source_smiles/fold_0')
    scores_reg_file = default_output / 'test_scores.csv'
    scores_clf_file = default_output / 'test_scores_clf.csv'
    history_file = default_output / 'training_history.json'
    model_file = default_output / 'best_model.pt'
    
    # Create trial directory
    trial_dir = search_results_dir / f'trial_{trial}'
    trial_dir.mkdir(exist_ok=True)
    
    # Initialize result dict
    result = hyper.copy()
    result.update({
        'trial': trial,
        'config_path': str(config_path.relative_to(Path('../results')))
    })
    
    # Process REGRESSION results
    if search_type in ['regression', 'both'] and scores_reg_file.exists():
        scores_df = pd.read_csv(scores_reg_file)
        avg_rmse = scores_df['RMSE'].mean()
        
        shutil.copy(scores_reg_file, trial_dir / 'test_scores.csv')
        if history_file.exists():
            shutil.copy(history_file, trial_dir / 'training_history.json')
        
        print(f"  Regression RMSE: {avg_rmse:.4f}")
        result['avg_rmse'] = avg_rmse
        
        if avg_rmse < best_regression['rmse']:
            best_regression.update({'rmse': avg_rmse, 'trial': trial, 'config': str(config_path)})
            if model_file.exists():
                shutil.copy(model_file, search_results_dir / 'best_regression_model.pt')
                shutil.copy(config_path, search_results_dir / 'best_regression_config.json')
                print(f"  ★ New best REGRESSION model! (RMSE: {avg_rmse:.4f})")
    else:
        result['avg_rmse'] = None
    
    # Process CLASSIFICATION results
    if search_type in ['classification', 'both'] and scores_clf_file.exists():
        clf_df = pd.read_csv(scores_clf_file)
        avg_auc = clf_df['AUC'].mean()
        avg_pr_auc = clf_df['PR_AUC'].mean()
        
        shutil.copy(scores_clf_file, trial_dir / 'test_scores_clf.csv')
        
        print(f"  Classification AUC: {avg_auc:.4f}, PR_AUC: {avg_pr_auc:.4f}")
        result['avg_auc'] = avg_auc
        result['avg_pr_auc'] = avg_pr_auc
        
        if avg_auc > best_classification['auc']:
            best_classification.update({'auc': avg_auc, 'trial': trial, 'config': str(config_path)})
            if model_file.exists():
                shutil.copy(model_file, search_results_dir / 'best_classification_model.pt')
                shutil.copy(config_path, search_results_dir / 'best_classification_config.json')
                print(f"  ★ New best CLASSIFICATION model! (AUC: {avg_auc:.4f})")
    else:
        result['avg_auc'] = None
        result['avg_pr_auc'] = None
    
    # Add result
    results.append(result)
    
    # Clean up default output directory
    if default_output.exists():
        shutil.rmtree(default_output)
    
    # Save intermediate results
    if results:
        results_df = pd.DataFrame(results)
        
        # Save all results
        results_df.to_csv(search_results_dir / 'search_results.csv', index=False)
        
        # Save regression-sorted results (by avg_rmse)
        if search_type in ['regression', 'both']:
            reg_df = results_df[results_df['avg_rmse'].notna()].sort_values('avg_rmse')
            reg_df.to_csv(search_results_dir / 'search_results_by_regression.csv', index=False)
        
        # Save classification-sorted results (by avg_auc)
        if search_type in ['classification', 'both']:
            clf_df = results_df[results_df['avg_auc'].notna()].sort_values('avg_auc', ascending=False)
            clf_df.to_csv(search_results_dir / 'search_results_by_classification.csv', index=False)

# Final summary
print(f"\n{'='*80}")
print("Hyperparameter Search Complete!")
print(f"{'='*80}")
print(f"Total trials completed: {len(results)}")

if search_type in ['regression', 'both'] and best_regression['trial'] is not None:
    print(f"\n{'='*80}")
    print("BEST REGRESSION MODEL:")
    print(f"{'='*80}")
    print(f"  Trial: {best_regression['trial']}")
    print(f"  Average RMSE: {best_regression['rmse']:.4f}")
    print(f"  Config: {best_regression['config']}")

if search_type in ['classification', 'both'] and best_classification['trial'] is not None:
    print(f"\n{'='*80}")
    print("BEST CLASSIFICATION MODEL:")
    print(f"{'='*80}")
    print(f"  Trial: {best_classification['trial']}")
    print(f"  Average AUC: {best_classification['auc']:.4f}")
    print(f"  Config: {best_classification['config']}")

print(f"\n{'='*80}")
print("OUTPUT FILES:")
print(f"{'='*80}")
print(f"  Search results directory: {search_results_dir}")
if search_type in ['regression', 'both']:
    print(f"  Regression results: search_results_regression.csv")
if search_type in ['classification', 'both']:
    print(f"  Classification results: search_results_classification.csv")
print(f"{'='*80}\n")