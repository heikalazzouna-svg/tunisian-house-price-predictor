import logging
import pandas as pd
import numpy as np
from zenml import step
from typing import Optional

logging.basicConfig(level=logging.INFO)

@step
def tunisian_outlier_detection_step(
    df: pd.DataFrame,
    method: str = "zscore",
    threshold: float = 3.0,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Detect and remove outliers from Tunisian housing data
    
    Args:
        df: Input dataframe
        method: Outlier detection method ('zscore' or 'iqr')
        threshold: Threshold for outlier detection
        columns: Specific columns to check for outliers (None = all numerical)
    
    Returns:
        Dataframe with outliers removed
    """
    logging.info(f"Starting outlier detection on Tunisian data")
    logging.info(f"Data shape before: {df.shape}")
    
    # Make a copy
    df_clean = df.copy()
    
    # Select columns to check for outliers
    if columns is None:
        # Use numerical columns except target and encoded columns
        exclude_cols = ['price_tnd', 'city_encoded', 'neighborhood_encoded', 
                       'building_type_encoded', 'price_per_m2']
        columns = [col for col in df_clean.select_dtypes(include=['float64', 'int64']).columns 
                  if col not in exclude_cols]
    
    logging.info(f"Checking for outliers in columns: {columns}")
    
    # Detect outliers
    outlier_mask = pd.Series(False, index=df_clean.index)
    
    for col in columns:
        if col in df_clean.columns:
            if method == "zscore":
                # Z-score method
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                col_outliers = z_scores > threshold
                outlier_mask = outlier_mask | col_outliers
                if col_outliers.sum() > 0:
                    logging.info(f"  {col}: {col_outliers.sum()} outliers detected")
                    
            elif method == "iqr":
                # IQR method
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                col_outliers = (df_clean[col] < Q1 - 1.5 * IQR) | (df_clean[col] > Q3 + 1.5 * IQR)
                outlier_mask = outlier_mask | col_outliers
                if col_outliers.sum() > 0:
                    logging.info(f"  {col}: {col_outliers.sum()} outliers detected")
    
    # Remove outliers
    outlier_count = outlier_mask.sum()
    if outlier_count > 0:
        df_clean = df_clean[~outlier_mask]
        logging.info(f"✅ Removed {outlier_count} outliers")
        logging.info(f"Data shape after: {df_clean.shape}")
    else:
        logging.info("✅ No outliers detected")
    
    return df_clean