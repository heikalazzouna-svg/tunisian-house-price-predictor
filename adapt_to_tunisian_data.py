import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def load_tunisian_data():
    """Load and prepare Tunisian housing data for the pipeline"""
    # Load the data
    df = pd.read_csv('data/tunisian_housing_data.csv')
    
    print(f"📊 Original data shape: {df.shape}")
    print(f"📊 Columns: {df.columns.tolist()}")
    
    # Prepare the data for the pipeline
    # Convert categorical variables to numerical using label encoding
    label_encoders = {}
    
    categorical_columns = ['city', 'neighborhood', 'building_type']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"✅ Encoded {col}: {len(le.classes_)} unique values")
    
    # Drop original categorical columns (keep encoded versions)
    df = df.drop(columns=categorical_columns)
    
    # Boolean columns to integers
    bool_columns = ['has_elevator', 'has_parking', 'has_terrace', 'has_garden', 'has_pool']
    for col in bool_columns:
        df[col] = df[col].astype(int)
    
    # Add some engineered features
    df['price_per_m2'] = df['price_tnd'] / df['surface_area_m2']
    df['rooms_per_surface'] = df['rooms'] / df['surface_area_m2']
    
    # Drop unnecessary columns
    df = df.drop(columns=['address', 'age'])
    
    print(f"\n✅ Data prepared with shape: {df.shape}")
    print(f"📊 Final columns: {df.columns.tolist()}")
    
    return df, label_encoders

# Load and prepare the data
print("🏠 Loading and preparing Tunisian housing data...")
df, encoders = load_tunisian_data()

# Split features and target
X = df.drop(columns=['price_tnd', 'price_per_m2'])
y = df['price_tnd']

# Create data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Save the prepared data
X.to_csv('data/X_features.csv', index=False)
y.to_csv('data/y_target.csv', index=False)

# Also save the full prepared dataset
df.to_csv('data/prepared_tunisian_data.csv', index=False)

print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"✅ Files saved to 'data/' folder")
print(f"   - X_features.csv: Feature matrix")
print(f"   - y_target.csv: Target values")
print(f"   - prepared_tunisian_data.csv: Full dataset")