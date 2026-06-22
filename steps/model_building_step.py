import logging
from typing import Annotated

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType

# Import MLflow utilities
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

logging.basicConfig(level=logging.INFO)


@step(enable_cache=False)
def model_building_step(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Annotated[
    Pipeline, 
    ArtifactConfig(
        name="sklearn_pipeline",
        artifact_type=ArtifactType.MODEL
    )
]:
    """
    Builds and trains a Random Forest model with MLflow tracking.
    """
    logging.info("=" * 60)
    logging.info("🚀 Starting model building step")
    
    # Set MLflow tracking URI from ZenML
    try:
        tracking_uri = get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        logging.info(f"✅ MLflow tracking URI: {tracking_uri}")
    except Exception as e:
        logging.warning(f"⚠️ Could not set MLflow tracking URI: {e}")
        # Fallback to default
        mlflow.set_tracking_uri("file:./mlruns")
    
    # Ensure inputs are correct
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns

    logging.info(f"📊 Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"📊 Numerical columns: {numerical_cols.tolist()}")

    # --- Preprocessing ---
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # --- Model with best parameters from tuning ---
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # --- Train with MLflow tracking ---
    # Start MLflow run explicitly
    with mlflow.start_run(run_name="model_training") as run:
        logging.info(f"🔍 MLflow Run ID: {run.info.run_id}")
        
        # Enable autologging
        mlflow.sklearn.autolog()
        
        # Log parameters BEFORE training
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth": "None",
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "numerical_features": len(numerical_cols),
            "categorical_features": len(categorical_cols)
        })
        logging.info("✅ Parameters logged to MLflow")

        # Train the model
        logging.info("🏋️ Training the model...")
        pipeline.fit(X_train, y_train)
        logging.info("✅ Model training completed")

        # Calculate and log training score
        train_score = pipeline.score(X_train, y_train)
        mlflow.log_metric("train_r2", train_score)
        logging.info(f"📈 Training R² Score: {train_score:.4f}")

        # Log feature importance
        try:
            feature_importance = pipeline.named_steps["model"].feature_importances_
            preprocessor = pipeline.named_steps["preprocessor"]
            feature_names = numerical_cols.tolist()
            if len(categorical_cols) > 0:
                onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
                onehot.fit(X_train[categorical_cols])
                feature_names.extend(onehot.get_feature_names_out(categorical_cols))
            
            # Log as artifact
            import json
            importance_dict = {name: float(imp) for name, imp in zip(feature_names, feature_importance)}
            with open("feature_importance.json", "w") as f:
                json.dump(importance_dict, f, indent=2)
            mlflow.log_artifact("feature_importance.json")
            logging.info("✅ Feature importance logged")
        except Exception as e:
            logging.warning(f"⚠️ Could not log feature importance: {e}")

        # Log the model
        mlflow.sklearn.log_model(pipeline, "model")
        logging.info("✅ Model logged to MLflow")
        
        # Save model URI for potential deployment
        model_uri = f"runs:/{run.info.run_id}/model"
        logging.info(f"🔗 Model URI: {model_uri}")

    logging.info("=" * 60)
    logging.info("✅ Model building step completed successfully")
    
    return pipeline