import click
import pandas as pd
import numpy as np
from zenml import pipeline, step
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

# Import your existing steps
from steps.handle_missing_values_step import handle_missing_values_step
from steps.model_building_step import model_building_step
from steps.model_evaluator_step import model_evaluator_step

# Import Tunisian-specific steps
from steps.tunisian_feature_engineering_step import tunisian_feature_engineering_step
from steps.tunisian_data_splitter_step import tunisian_data_splitter_step
from steps.tunisian_outlier_detection_step import tunisian_outlier_detection_step

@step
def real_tunisian_data_ingestion_step() -> pd.DataFrame:
    """Load Real Tunisian housing data"""
    df = pd.read_csv('data/real_tunisian_data_adapted.csv')
    print(f"✅ Loaded Real Tunisian data with shape: {df.shape}")
    print(f"📊 Columns: {df.columns.tolist()}")
    return df

@pipeline
def real_tunisian_ml_pipeline():
    """ML Pipeline for Real Tunisian housing price prediction"""
    
    print("\n🏠 Running Real Tunisian ML Pipeline")
    print("=" * 50)
    
    # Data ingestion
    df = real_tunisian_data_ingestion_step()
    
    # Handle missing values
    df = handle_missing_values_step(df)
    
    # Feature engineering
    df = tunisian_feature_engineering_step(df)
    
    # Outlier detection
    df = tunisian_outlier_detection_step(df)
    
    # Data split
    X_train, X_test, y_train, y_test = tunisian_data_splitter_step(df)
    
    # Model building
    model = model_building_step(X_train=X_train, y_train=y_train)
    
    # Model evaluation
    model_evaluator_step(trained_model=model, X_test=X_test, y_test=y_test)

@click.command()
def main():
    """Run the Real Tunisian ML pipeline"""
    
    print("🏠 Real Tunisian Housing Price Prediction Pipeline")
    print("=" * 50)
    
    # Set environment variable for caching
    import os
    os.environ["ZENML_DISABLE_CACHE"] = "true"
    
    # Run the pipeline
    run = real_tunisian_ml_pipeline()
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📊 Run ID: {run.id}")
    
    # Get tracking URI
    try:
        tracking_uri = get_tracking_uri()
        print(f"\n🔗 MLflow tracking available at: {tracking_uri}")
        print("Run 'mlflow ui' to view results")
    except:
        print("\n🔗 MLflow tracking available")
        print("Run 'mlflow ui' to view results")

if __name__ == "__main__":
    main()