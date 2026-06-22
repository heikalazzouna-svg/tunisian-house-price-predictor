import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def adapt_real_tunisian_data():
    """
    Adapt the Real Tunisian housing data (merged from Tayara, Menzili, Mubawab)
    """
    print("🏠 Adapting Real Tunisian Housing Data (Merged Dataset)")
    print("=" * 60)
    
    # Load the merged dataset
    df = pd.read_csv('data/real_tunisian_data/mubawab_tayara_menzili_final.csv')
    print(f"✅ Loaded merged data shape: {df.shape}")
    print(f"📊 Columns: {df.columns.tolist()}")
    
    # --- CREATE CLEAN TARGET ---
    # Price is already in 'prix' column
    df['price_tnd'] = df['prix']
    print(f"✅ Target column set: prix → price_tnd")
    
    # --- RENAME COLUMNS TO MATCH PIPELINE ---
    column_mapping = {
        'superficie': 'surface_area_m2',
        'chambres': 'rooms',
        'salle_de_bains': 'bathrooms',
        'etage': 'floor_level',
        'nb_pieces': 'total_rooms',
    }
    
    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
            print(f"  ✅ Renamed: {old_name} → {new_name}")
    
    # --- HANDLE GOVERNORATE COLUMNS ---
    # The dataset has one-hot encoded governorate columns
    # Let's convert them back to a single 'city' column
    
    governorate_cols = ['gouvernorat_ariana', 'gouvernorat_ben-arous', 
                       'gouvernorat_la-manouba', 'gouvernorat_tunis']
    
    # Create city column from the one-hot encoded columns
    def get_city(row):
        for col in governorate_cols:
            if row[col] == 1:
                # Extract city name from column name
                return col.replace('gouvernorat_', '')
        return 'other'
    
    df['city'] = df.apply(get_city, axis=1)
    print(f"  ✅ Created 'city' column from governorate columns")
    
    # Drop the one-hot encoded columns
    df = df.drop(columns=governorate_cols)
    print(f"  ✅ Dropped governorate one-hot columns")
    
    # --- ENCODE CITY COLUMN ---
    le = LabelEncoder()
    df['city_encoded'] = le.fit_transform(df['city'])
    print(f"  ✅ Encoded city: {len(le.classes_)} unique cities")
    
    # --- CREATE ADDITIONAL FEATURES ---
    # Price per square meter
    if 'surface_area_m2' in df.columns:
        df['price_per_m2'] = df['price_tnd'] / df['surface_area_m2']
        print("  ✅ Added price_per_m2")
    
    # Rooms per surface
    if 'surface_area_m2' in df.columns and 'rooms' in df.columns:
        df['rooms_per_surface'] = df['rooms'] / df['surface_area_m2']
        print("  ✅ Added rooms_per_surface")
    
    # --- DROP UNNECESSARY COLUMNS ---
    drop_cols = ['delegation', 'nb_pieces']
    drop_existing = [col for col in drop_cols if col in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)
        print(f"  ✅ Dropped: {drop_existing}")
    
    # --- HANDLE MISSING VALUES ---
    print("\n🔧 Handling missing values...")
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  ✅ Filled {col} with median: {median_val:.2f}")
    
    # --- SAVE ADAPTED DATA ---
    output_path = 'data/real_tunisian_data_adapted.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Data adapted and saved to: {output_path}")
    
    print(f"\n📊 Final shape: {df.shape}")
    print(f"📊 Final columns: {df.columns.tolist()}")
    
    print("\n📋 Sample data:")
    print(df.head())
    
    print(f"\n📊 Price statistics:")
    print(df['price_tnd'].describe())
    
    return df

if __name__ == "__main__":
    adapt_real_tunisian_data()