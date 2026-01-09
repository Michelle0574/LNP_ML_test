import pandas as pd
import numpy as np
import re
from typing import Tuple


def process_pdi_column(pdi_series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Process PDI column from format "0.2 ~ 0.3" to [mid, width, is_interval]
    # Returns: (pdi_mid, pdi_width, pdi_is_interval)
    
    pdi_mid = []
    pdi_width = []
    pdi_is_interval = []
    
    for val in pdi_series:
        if pd.isna(val):
            pdi_mid.append(0.0)
            pdi_width.append(0.0)
            pdi_is_interval.append(0.0)
            continue
        
        val_str = str(val).strip()
        
        # Check if it's an interval (contains ~)
        if '~' in val_str:
            parts = val_str.split('~')
            if len(parts) == 2:
                try:
                    low = float(parts[0].strip())
                    high = float(parts[1].strip())
                    mid = (low + high) / 2.0
                    width = high - low
                    is_interval = 1.0
                except:
                    mid = 0.0
                    width = 0.0
                    is_interval = 0.0
            else:
                mid = 0.0
                width = 0.0
                is_interval = 0.0
        else:
            # Single value
            try:
                mid = float(val_str)
                width = 0.0
                is_interval = 0.0
            except:
                mid = 0.0
                width = 0.0
                is_interval = 0.0
        
        pdi_mid.append(mid)
        pdi_width.append(width)
        pdi_is_interval.append(is_interval)
    
    return (
        pd.Series(pdi_mid, index=pdi_series.index),
        pd.Series(pdi_width, index=pdi_series.index),
        pd.Series(pdi_is_interval, index=pdi_series.index)
    )

def process_categorical_columns(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Dynamically create one-hot encodings for categorical columns.
    This is called during data loading, not saved to CSV.
    """
    df_processed = df.copy()
    
    # 1. Process Purity
    if 'Purity' in df.columns:
        if 'Purity_Pure' not in df.columns:
            df_processed['Purity_Pure'] = (df['Purity'] == 'Pure').astype(int)
        if 'Purity_Crude' not in df.columns:
            df_processed['Purity_Crude'] = (df['Purity'] == 'Crude').astype(int)
    
    # 2. Process Mix_type
    if 'Mix_type' in df.columns:
        if 'Mix_type_Hand' not in df.columns:
            df_processed['Mix_type_Hand'] = (df['Mix_type'] == 'Hand').astype(int)
        if 'Mix_type_Microfluidic' not in df.columns:
            df_processed['Mix_type_Microfluidic'] = (df['Mix_type'] == 'Microfluidic').astype(int)
    
    # 3. Process toxic column (convert to 0/1)
    if 'toxic' in df.columns:
        df_processed['toxic'] = pd.to_numeric(df['toxic'], errors='coerce')
        df_processed['toxic'] = df_processed['toxic'].fillna(0).astype(int)
        df_processed['toxic'] = df_processed['toxic'].apply(lambda x: 1 if x > 0 else 0)
    
    # 4. Process Target_or_delivered_gene
    if 'Target_or_delivered_gene' in df.columns:
        unique_genes = df['Target_or_delivered_gene'].dropna().unique()
        print(f"[Preprocessing] Found {len(unique_genes)} unique Target_or_delivered_gene values: {sorted(unique_genes)}")
        
        for gene in unique_genes:
            col_name = f'Target_or_delivered_gene_{gene}'
            if col_name not in df_processed.columns:
                df_processed[col_name] = (df['Target_or_delivered_gene'] == gene).astype(int)
        
        # Add a column for "no gene specified" (NaN values)
        if 'Target_or_delivered_gene_None' not in df_processed.columns:
            df_processed['Target_or_delivered_gene_None'] = df['Target_or_delivered_gene'].isna().astype(int)
    
    # 5. Process Value_name
    if 'Value_name' in df.columns:
        unique_values = df['Value_name'].dropna().unique()
        print(f"[Preprocessing] Found {len(unique_values)} unique Value_name values: {sorted(unique_values)}")
        
        for value_name in unique_values:
            col_name = f'Value_name_{value_name}'
            if col_name not in df_processed.columns:
                df_processed[col_name] = (df['Value_name'] == value_name).astype(int)
        
        # Add a column for "no value name specified" (NaN values)
        if 'Value_name_None' not in df_processed.columns:
            df_processed['Value_name_None'] = df['Value_name'].isna().astype(int)
    
    return df_processed


def preprocess_dataframe(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Apply all preprocessing: PDI, Purity, categorical encoding, etc.
    """
    df_processed = df.copy()
    
    # 1. Process categorical columns FIRST (NEW)
    df_processed = process_categorical_columns(df_processed, config)
    
    # 2. Process PDI if enabled
    if config.pdi_enabled and 'PDI' in df.columns:
        print("[Preprocessing] Processing PDI column...")
        pdi_mid, pdi_width, pdi_is_interval = process_pdi_column(df['PDI'])
        df_processed['pdi_mid'] = pdi_mid
        df_processed['pdi_width'] = pdi_width
        df_processed['pdi_is_interval'] = pdi_is_interval
        print(f"  PDI -> [pdi_mid, pdi_width, pdi_is_interval]")
        print(f"  Sample: {df['PDI'].iloc[0]} -> [{pdi_mid.iloc[0]:.3f}, {pdi_width.iloc[0]:.3f}, {pdi_is_interval.iloc[0]:.0f}]")
    
    # 3. REMOVED: Purity processing (now handled in process_categorical_columns)
    # The old process_purity_column is no longer needed
    
    return df_processed

def process_purity_column(df: pd.DataFrame, purity_col: str = 'Purity') -> pd.DataFrame:
    # Convert Purity column to one-hot encoding (Pure, Crude)
    # Returns: dataframe with Purity_Pure and Purity_Crude columns
    
    if purity_col not in df.columns:
        return df
    
    # Get unique values and create one-hot
    df_copy = df.copy()
    purity_dummies = pd.get_dummies(df_copy[purity_col], prefix='Purity')
    
    # Ensure both columns exist
    if 'Purity_Pure' not in purity_dummies.columns:
        purity_dummies['Purity_Pure'] = 0
    if 'Purity_Crude' not in purity_dummies.columns:
        purity_dummies['Purity_Crude'] = 0
    
    # Add to dataframe
    df_copy = pd.concat([df_copy, purity_dummies[['Purity_Pure', 'Purity_Crude']]], axis=1)
    
    return df_copy

