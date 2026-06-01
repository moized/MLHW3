"""
Preprocessing Module
Handles skewness calculation, log transformation, and standardization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


def calculate_skewness(data: np.ndarray) -> float:
    """
    Calculate Fisher-Pearson skewness coefficient (g1).
    
    Formula: g1 = (1/n) * Σ((xi - μ) / σ)³
    
    Interpretation:
    - g1 = 0: Normal (symmetric) distribution
    - g1 > 0: Right-skewed (positive skew)
    - g1 < 0: Left-skewed (negative skew)
    
    Args:
        data: 1D numpy array of values
        
    Returns:
        Skewness coefficient (g1)
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=0)  # Population std
    
    if std == 0:
        return 0.0
    
    # Fisher-Pearson coefficient
    g1 = np.mean(((data - mean) / std) ** 3)
    
    return g1


def calculate_all_skewness(rfm_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate skewness for all RFM features.
    
    Args:
        rfm_df: DataFrame with RFM values
        
    Returns:
        Dictionary with feature names and their skewness values
    """
    features = ['Recency', 'Frequency', 'Monetary']
    skewness_values = {}
    
    print("\n" + "=" * 50)
    print("SKEWNESS ANALYSIS (Fisher-Pearson Coefficient)")
    print("=" * 50)
    
    for feature in features:
        g1 = calculate_skewness(rfm_df[feature].values)
        skewness_values[feature] = g1
        
        # Interpret skewness
        if abs(g1) < 0.5:
            interpretation = "Approximately symmetric"
        elif g1 > 0:
            interpretation = "Right-skewed (positive)"
        else:
            interpretation = "Left-skewed (negative)"
            
        print(f"{feature:12s}: g1 = {g1:8.4f} ({interpretation})")
    
    return skewness_values


def apply_log_transform(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log-plus-one transformation: y = ln(1 + x)
    
    This transformation helps normalize right-skewed distributions.
    
    Args:
        rfm_df: DataFrame with RFM values
        
    Returns:
        DataFrame with log-transformed RFM values
    """
    print("\n" + "=" * 50)
    print("APPLYING LOG TRANSFORMATION: y = ln(1 + x)")
    print("=" * 50)
    
    rfm_transformed = rfm_df.copy()
    features = ['Recency', 'Frequency', 'Monetary']
    
    for feature in features:
        original_min = rfm_transformed[feature].min()
        original_max = rfm_transformed[feature].max()
        
        rfm_transformed[feature] = np.log1p(rfm_transformed[feature])
        
        new_min = rfm_transformed[feature].min()
        new_max = rfm_transformed[feature].max()
        
        print(f"{feature:12s}: [{original_min:.2f}, {original_max:.2f}] -> [{new_min:.4f}, {new_max:.4f}]")
    
    return rfm_transformed


def apply_standardization(rfm_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply z-score standardization to RFM features.
    
    Formula: z = (x - μ) / σ
    
    Args:
        rfm_df: DataFrame with RFM values (possibly log-transformed)
        
    Returns:
        Tuple of (standardized DataFrame, fitted scaler)
    """
    print("\n" + "=" * 50)
    print("APPLYING STANDARDIZATION (Z-Score Normalization)")
    print("=" * 50)
    
    features = ['Recency', 'Frequency', 'Monetary']
    
    scaler = StandardScaler()
    rfm_scaled = rfm_df.copy()
    rfm_scaled[features] = scaler.fit_transform(rfm_df[features])
    
    print(f"Features standardized to μ=0, σ=1")
    print(f"\nScaler parameters:")
    for i, feature in enumerate(features):
        print(f"  {feature:12s}: mean = {scaler.mean_[i]:.4f}, std = {scaler.scale_[i]:.4f}")
    
    return rfm_scaled, scaler


def preprocess_rfm(rfm_df: pd.DataFrame, apply_log: bool = True) -> Tuple[pd.DataFrame, StandardScaler, Dict]:
    """
    Complete preprocessing pipeline for RFM data.
    
    Steps:
    1. Calculate initial skewness
    2. Apply log transformation (if needed)
    3. Calculate skewness after transformation
    4. Apply standardization
    
    Args:
        rfm_df: DataFrame with raw RFM values
        apply_log: Whether to apply log transformation
        
    Returns:
        Tuple of (preprocessed DataFrame, scaler, preprocessing info dict)
    """
    info = {}
    
    # Step 1: Calculate initial skewness
    print("\n" + "#" * 60)
    print("STEP 1: INITIAL SKEWNESS ANALYSIS")
    print("#" * 60)
    info['initial_skewness'] = calculate_all_skewness(rfm_df)
    
    # Count features with non-zero skewness
    non_zero_skew = sum(1 for v in info['initial_skewness'].values() if abs(v) > 0.1)
    print(f"\nFeatures with significant skewness: {non_zero_skew}/3")
    
    # Step 2: Apply log transformation if at least 2 features are skewed
    if apply_log and non_zero_skew >= 2:
        print("\n" + "#" * 60)
        print("STEP 2: LOG TRANSFORMATION")
        print("#" * 60)
        rfm_transformed = apply_log_transform(rfm_df)
        info['log_applied'] = True
        
        # Step 3: Calculate skewness after transformation
        print("\n" + "#" * 60)
        print("STEP 3: SKEWNESS AFTER TRANSFORMATION")
        print("#" * 60)
        info['transformed_skewness'] = calculate_all_skewness(rfm_transformed)
    else:
        rfm_transformed = rfm_df.copy()
        info['log_applied'] = False
        info['transformed_skewness'] = info['initial_skewness']
    
    # Step 4: Apply standardization
    print("\n" + "#" * 60)
    print("STEP 4: STANDARDIZATION")
    print("#" * 60)
    rfm_scaled, scaler = apply_standardization(rfm_transformed)
    
    return rfm_scaled, scaler, info


if __name__ == "__main__":
    # Test the module
    from dataset import get_rfm_data
    
    rfm_df = get_rfm_data()
    rfm_processed, scaler, info = preprocess_rfm(rfm_df)
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Log transformation applied: {info['log_applied']}")
    print("\nProcessed RFM sample:")
    print(rfm_processed.head())
