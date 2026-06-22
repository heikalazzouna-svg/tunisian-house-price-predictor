import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

class TunisianDataIngestor:
    """Data ingestor specifically for Tunisian housing data"""
    
    def __init__(self):
        self.label_encoders = {}
    
    def ingest(self, file_path=None):
        """Load and prepare Tunisian housing data"""
        if file_path is None:
            file_path = 'data/prepared_tunisian_data.csv'
        
        # Load the data
        df = pd.read_csv(file_path)
        
        print(f"✅ Loaded Tunisian data with shape: {df.shape}")
        print(f"📊 Columns: {df.columns.tolist()}")
        
        return df
    
    def prepare_features(self, df):
        """Separate features and target"""
        # Target is 'price_tnd'
        if 'price_tnd' in df.columns:
            X = df.drop(columns=['price_tnd', 'price_per_m2'], errors='ignore')
            y = df['price_tnd']
            return X, y
        else:
            return df
    
    def get_feature_names(self):
        """Return list of feature names"""
        df = pd.read_csv('data/prepared_tunisian_data.csv')
        X, _ = self.prepare_features(df)
        return X.columns.tolist()

# Example usage
if __name__ == "__main__":
    ingestor = TunisianDataIngestor()
    data = ingestor.ingest()
    X, y = ingestor.prepare_features(data)
    
    print(f"\n✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    print(f"📋 Feature names: {X.columns.tolist()[:10]}...")