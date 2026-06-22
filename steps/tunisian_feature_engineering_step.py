import logging
import pandas as pd
import numpy as np
from zenml import step
from typing import List, Optional

logging.basicConfig(level=logging.INFO)

@step
def tunisian_feature_engineering_step(
    df: pd.DataFrame,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Feature engineering step specifically for Tunisian housing data
    
    Args:
        df: Input dataframe
        features: List of features to transform (optional)
    """
    logging.info("Applying feature engineering to Tunisian data")
    
    # Make a copy to avoid modifying the original
    df_transformed = df.copy()
    
    # If no features specified, use numerical columns
    if features is None:
        # Identify numerical columns for log transformation
        numerical_cols = df_transformed.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Remove target and ID-like columns
        exclude_cols = ['price_tnd', 'price_per_m2', 'year_built', 'city_encoded', 
                       'neighborhood_encoded', 'building_type_encoded']
        features = [col for col in numerical_cols if col not in exclude_cols]
        
        logging.info(f"Auto-selected features for transformation: {features}")
    
    # Log transform skewed features (using log1p to handle zeros)
    for feature in features:
        if feature in df_transformed.columns:
            # Check if feature has positive values
            if (df_transformed[feature] > 0).all():
                df_transformed[f'{feature}_log'] = np.log1p(df_transformed[feature])
                logging.info(f"✅ Log transformed: {feature}")
            else:
                logging.warning(f"⚠️ Skipping {feature} - contains non-positive values")
    
    # Add interaction features
    if 'surface_area_m2' in df_transformed.columns and 'rooms' in df_transformed.columns:
        df_transformed['rooms_per_100sqm'] = df_transformed['rooms'] / (df_transformed['surface_area_m2'] / 100)
        logging.info("✅ Added rooms_per_100sqm feature")
    
    if 'has_parking' in df_transformed.columns and 'has_garden' in df_transformed.columns:
        df_transformed['has_amenities'] = (df_transformed['has_parking'] + df_transformed['has_garden']).astype(int)
        logging.info("✅ Added has_amenities feature")
    
    # Add age feature if year_built exists
    if 'year_built' in df_transformed.columns:
        df_transformed['property_age'] = 2024 - df_transformed['year_built']
        logging.info("✅ Added property_age feature")
    
    logging.info(f"✅ Feature engineering completed. Shape: {df_transformed.shape}")
    logging.info(f"📊 New columns: {df_transformed.columns.tolist()}")
    
    return df_transformed