import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from zenml import step
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)

@step
def tunisian_data_splitter_step(
    df: pd.DataFrame,
    test_size: Optional[float] = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split Tunisian housing data into train and test sets
    
    Args:
        df: Input dataframe
        test_size: Proportion of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
    """
    logging.info("Splitting Tunisian housing data")
    logging.info(f"Data shape: {df.shape}")
    
    # Check if price_tnd exists
    if 'price_tnd' not in df.columns:
        raise ValueError("Target column 'price_tnd' not found in dataframe")
    
    # Separate target and features
    X = df.drop(columns=['price_tnd'])
    y = df['price_tnd']
    
    logging.info(f"Features shape: {X.shape}")
    logging.info(f"Target shape: {y.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logging.info(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logging.info(f"✅ Train target shape: {y_train.shape}, Test target shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test