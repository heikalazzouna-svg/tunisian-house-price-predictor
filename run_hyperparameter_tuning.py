import click
import pandas as pd
import numpy as np
from typing import Tuple
from zenml import pipeline, step
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from sklearn.model_selection import train_test_split

# Import steps
from steps.hyperparameter_tuning_step import hyperparameter_tuning_step
from steps.model_evaluator_step import model_evaluator_step


@step
def load_real_tunisian_data() -> pd.DataFrame:
    """Load Real Tunisian housing data"""
    df = pd.read_csv('data/real_tunisian_data_adapted.csv')
    print(f"✅ Loaded data with shape: {df.shape}")
    print(f"📊 Columns: {df.columns.tolist()}")
    return df


@step
def prepare_data_for_tuning(df: pd.DataFrame) -> Tuple[
    pd.DataFrame,  # X_train
    pd.DataFrame,  # X_test
    pd.Series,     # y_train
    pd.Series      # y_test
]:
    """Prepare data for tuning - split into train and test"""
    print(f"🔍 Preparing data for tuning...")
    print(f"Data shape: {df.shape}")
    
    # Check if 'price_tnd' exists
    if 'price_tnd' not in df.columns:
        print(f"⚠️ 'price_tnd' not found! Available columns: {df.columns.tolist()}")
        if 'prix' in df.columns:
            df['price_tnd'] = df['prix']
            print("✅ Created 'price_tnd' from 'prix'")
        else:
            raise ValueError("No target column found! Looking for 'price_tnd' or 'prix'")
    
    # Split features and target
    X = df.drop(columns=['price_tnd'])
    y = df['price_tnd']
    
    print(f"✅ Features shape: {X.shape}")
    print(f"✅ Target shape: {y.shape}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✅ Train shape: {X_train.shape}")
    print(f"✅ Test shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


@pipeline
def tuning_pipeline():
    """Pipeline for hyperparameter tuning"""
    
    print("\n🔍 Hyperparameter Tuning Pipeline")
    print("=" * 50)
    
    # Load data
    df = load_real_tunisian_data()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_for_tuning(df)
    
    # Run hyperparameter tuning
    best_model, best_params = hyperparameter_tuning_step(
        X_train=X_train,
        y_train=y_train,
        model_type="random_forest",
        search_type="grid",
        cv_folds=3
    )
    
    print(f"\n✅ Best Model: {best_model}")
    print(f"✅ Best Parameters: {best_params}")
    
    # Optional: Evaluate the best model on test data
    # model_evaluator_step(trained_model=best_model, X_test=X_test, y_test=y_test)


@click.command()
def main():
    """Run hyperparameter tuning"""
    import os
    os.environ["ZENML_DISABLE_CACHE"] = "true"
    
    run = tuning_pipeline()
    print(f"\n✅ Pipeline completed! Run ID: {run.id}")


if __name__ == "__main__":
    main()