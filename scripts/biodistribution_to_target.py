"""
Biodistribution to Delivery Target Conversion
Converts Biodistribution regression predictions to Delivery_target binary classifications
using configurable organ-specific thresholds
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


def biodistribution_to_delivery_targets(
    biodist_df: pd.DataFrame, 
    threshold_type: str = 'percentile',
    threshold_value: float = 0.3,
    min_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Convert Biodistribution predictions to multi-label Delivery_target classifications
    
    Args:
        biodist_df (pd.DataFrame): DataFrame containing Biodistribution_* columns
        threshold_type (str): Threshold calculation method
            - 'percentile': Use percentile-based threshold (e.g., top 30%)
            - 'absolute': Use fixed absolute threshold (e.g., > 0.1)
            - 'relative': Use relative threshold (e.g., > 50% of max per sample)
        threshold_value (float): Threshold parameter (interpretation depends on threshold_type)
        min_threshold (float): Minimum absolute threshold to avoid noise
    
    Returns:
        pd.DataFrame: Input dataframe with added Delivery_target_* binary columns
    
    Example:
        >>> df = pd.DataFrame({'Biodistribution_liver': [0.4, 0.2, 0.1]})
        >>> result = biodistribution_to_delivery_targets(df, 'absolute', 0.25)
        >>> print(result['Delivery_target_liver'])
        [1.0, 0.0, 0.0]
    """
    # Organ mapping: Biodistribution columns -> Delivery_target columns
    organ_mapping = {
        'Biodistribution_liver': 'Delivery_target_liver',
        'Biodistribution_lung': 'Delivery_target_lung',
        'Biodistribution_spleen': 'Delivery_target_spleen',
        'Biodistribution_muscle': 'Delivery_target_muscle',
        'Biodistribution_lymph_nodes': 'Delivery_target_lymph_nodes',
        'Biodistribution_heart': 'Delivery_target_heart',
        'Biodistribution_kidney': 'Delivery_target_kidney',
    }
    
    result_df = biodist_df.copy()
    
    # Get all Biodistribution columns in the dataframe
    biodist_cols = [col for col in biodist_df.columns if col.startswith('Biodistribution_')]
    
    for biodist_col in biodist_cols:
        # Skip columns not in mapping
        if biodist_col not in organ_mapping:
            continue
            
        target_col = organ_mapping[biodist_col]
        values = biodist_df[biodist_col].values
        
        # Calculate threshold based on type
        if threshold_type == 'percentile':
            # Use percentile (e.g., top 30% of values)
            valid_values = values[~np.isnan(values)]
            if len(valid_values) > 0:
                threshold = np.percentile(valid_values, 100 - threshold_value * 100)
            else:
                threshold = min_threshold
                
        elif threshold_type == 'absolute':
            # Use fixed absolute threshold
            threshold = threshold_value
            
        elif threshold_type == 'relative':
            # Use relative threshold (handled per-sample below)
            threshold = None
            
        else:
            raise ValueError(f"Unknown threshold_type: {threshold_type}")
        
        # Generate binary labels
        if threshold_type == 'relative':
            # Relative threshold: per-sample, based on max biodistribution
            targets = np.zeros(len(values))
            for i, val in enumerate(values):
                if np.isnan(val):
                    continue
                row_biodist = biodist_df.iloc[i][biodist_cols].values
                row_max = np.nanmax(row_biodist)
                # Target if: value >= threshold_value * max AND value >= min_threshold
                if row_max > 0 and val >= threshold_value * row_max and val >= min_threshold:
                    targets[i] = 1.0
            result_df[target_col] = targets
        else:
            # Absolute or percentile threshold
            result_df[target_col] = np.where(
                (values >= threshold) & (values >= min_threshold) & (~np.isnan(values)),
                1.0,
                0.0
            )
    
    return result_df


def apply_organ_specific_thresholds(
    biodist_df: pd.DataFrame,
    organ_thresholds: Dict[str, float]
) -> pd.DataFrame:
    """
    Apply organ-specific thresholds for Biodistribution->Delivery_target conversion
    
    This is the recommended method for production use, as it allows fine-tuned
    control over each organ's threshold based on biological/experimental knowledge.
    
    Args:
        biodist_df (pd.DataFrame): DataFrame with Biodistribution_* columns
        organ_thresholds (dict): Mapping of organ names to threshold values
            Example: {'liver': 0.25, 'lung': 0.20, 'spleen': 0.15}
    
    Returns:
        pd.DataFrame: Input dataframe with added Delivery_target_* binary columns
    
    Example:
        >>> from postprocessing_config import get_organ_thresholds
        >>> thresholds = get_organ_thresholds('default')
        >>> result = apply_organ_specific_thresholds(df, thresholds)
    """
    result_df = biodist_df.copy()
    
    for organ, threshold in organ_thresholds.items():
        biodist_col = f'Biodistribution_{organ}'
        target_col = f'Delivery_target_{organ}'
        
        if biodist_col in biodist_df.columns:
            values = biodist_df[biodist_col].values
            # Binary classification: 1 if biodistribution >= threshold, else 0
            result_df[target_col] = np.where(
                (values >= threshold) & (~np.isnan(values)),
                1.0,
                0.0
            )
    
    return result_df


def apply_postprocessing_to_predictions(
    predictions_df: pd.DataFrame,
    organ_thresholds: Optional[Dict[str, float]] = None,
    prediction_prefix: str = '',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Apply post-processing to model predictions with flexible column naming
    
    This function handles predictions with various column naming conventions:
    - Direct: 'Biodistribution_liver'
    - With prefix: 'cv_0_pred_Biodistribution_liver'
    - Custom prefix: '{prediction_prefix}Biodistribution_liver'
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing prediction columns
        organ_thresholds (dict, optional): Organ-specific thresholds
            If None, uses default thresholds from config
        prediction_prefix (str): Prefix for prediction columns (e.g., 'cv_0_pred_')
        inplace (bool): If True, modify dataframe in-place
    
    Returns:
        pd.DataFrame: DataFrame with added Delivery_target prediction columns
    
    Example:
        >>> # For Chemprop predictions with 'cv_0_pred_' prefix
        >>> df = apply_postprocessing_to_predictions(
        ...     predictions_df,
        ...     organ_thresholds={'liver': 0.30, 'lung': 0.25},
        ...     prediction_prefix='cv_0_pred_'
        ... )
    """
    if organ_thresholds is None:
        from postprocessing_config import get_organ_thresholds
        organ_thresholds = get_organ_thresholds('default')
    
    if not inplace:
        result_df = predictions_df.copy()
    else:
        result_df = predictions_df
    
    # Find Biodistribution prediction columns
    if prediction_prefix:
        biodist_pred_cols = [
            col for col in result_df.columns 
            if col.startswith(prediction_prefix) and 'Biodistribution_' in col
        ]
    else:
        biodist_pred_cols = [
            col for col in result_df.columns 
            if col.startswith('Biodistribution_')
        ]
    
    if len(biodist_pred_cols) == 0:
        print(f"Warning: No Biodistribution prediction columns found with prefix '{prediction_prefix}'")
        return result_df
    
    # Create temporary dataframe with renamed columns for processing
    temp_df = result_df.copy()
    rename_dict = {}
    
    for col in biodist_pred_cols:
        # Remove prefix to get standard column name
        if prediction_prefix:
            standard_name = col.replace(prediction_prefix, '')
        else:
            standard_name = col
        rename_dict[col] = standard_name
    
    temp_df.rename(columns=rename_dict, inplace=True)
    
    # Apply thresholds
    temp_df_with_targets = apply_organ_specific_thresholds(temp_df, organ_thresholds)
    
    # Add generated Delivery_target columns back to result with prefix
    target_cols = [
        col for col in temp_df_with_targets.columns 
        if col.startswith('Delivery_target_')
    ]
    
    for target_col in target_cols:
        if prediction_prefix:
            new_col_name = prediction_prefix + target_col
        else:
            new_col_name = target_col
        result_df[new_col_name] = temp_df_with_targets[target_col]
    
    print(f"[Post-processing] Added {len(target_cols)} Delivery_target columns")
    
    return result_df


# Usage examples and testing
if __name__ == '__main__':
    print("\n" + "="*80)
    print("Testing Biodistribution to Delivery_target Conversion")
    print("="*80 + "\n")
    
    # Create sample data
    sample_data = {
        'smiles': ['CCO', 'CC', 'C'],
        'Biodistribution_liver': [0.35, 0.25, 0.15],
        'Biodistribution_lung': [0.28, 0.22, 0.10],
        'Biodistribution_spleen': [0.18, 0.15, 0.12],
    }
    df = pd.DataFrame(sample_data)
    
    print("Sample Input Data:")
    print(df)
    print()
    
    # Method 1: Using default thresholds
    print("\nMethod 1: Using default thresholds from config")
    from postprocessing_config import get_organ_thresholds
    thresholds = get_organ_thresholds('default')
    result1 = apply_organ_specific_thresholds(df, thresholds)
    print(result1[['Biodistribution_liver', 'Delivery_target_liver']])
    
    # Method 2: Using percentile-based threshold
    print("\nMethod 2: Using percentile-based threshold (top 30%)")
    result2 = biodistribution_to_delivery_targets(
        df,
        threshold_type='percentile',
        threshold_value=0.3,
        min_threshold=0.1
    )
    print(result2[['Biodistribution_liver', 'Delivery_target_liver']])
    
    # Method 3: With prediction prefix (simulating Chemprop output)
    print("\nMethod 3: With prediction prefix (cv_0_pred_)")
    df_with_prefix = df.rename(columns={
        'Biodistribution_liver': 'cv_0_pred_Biodistribution_liver',
        'Biodistribution_lung': 'cv_0_pred_Biodistribution_lung',
        'Biodistribution_spleen': 'cv_0_pred_Biodistribution_spleen',
    })
    result3 = apply_postprocessing_to_predictions(
        df_with_prefix,
        organ_thresholds=thresholds,
        prediction_prefix='cv_0_pred_'
    )
    print(result3[[col for col in result3.columns if 'liver' in col]])
    
    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80 + "\n")