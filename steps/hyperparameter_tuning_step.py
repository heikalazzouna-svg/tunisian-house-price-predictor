import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from zenml import step
from typing import Tuple, Dict, Any
import mlflow

logging.basicConfig(level=logging.INFO)


@step(enable_cache=False)
def hyperparameter_tuning_step(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    search_type: str = "grid",
    cv_folds: int = 3
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Perform hyperparameter tuning to find the best model parameters.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: 'random_forest', 'gradient_boosting', 'ridge', 'lasso'
        search_type: 'grid' or 'random'
        cv_folds: Number of cross-validation folds
    
    Returns:
        Tuple of (best_pipeline, best_params)
    """
    logging.info(f"🔍 Starting Hyperparameter Tuning")
    logging.info(f"Model Type: {model_type}")
    logging.info(f"Search Type: {search_type}")
    logging.info(f"CV Folds: {cv_folds}")
    
    # Identify categorical and numerical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns
    numerical_cols = X_train.select_dtypes(exclude=["object", "category"]).columns
    
    logging.info(f"Categorical columns: {categorical_cols.tolist()}")
    logging.info(f"Numerical columns: {numerical_cols.tolist()}")
    
    # --- Define Preprocessing ---
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    
    # Numerical transformer
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical transformer (only if there are categorical columns)
    if len(categorical_cols) > 0:
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
                ("cat", categorical_transformer, categorical_cols),
            ]
        )
    else:
        # Only numerical columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_cols),
            ]
        )
    
    # --- Define Models and Parameter Grids ---
    if model_type == "random_forest":
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, 15, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__max_features': ['sqrt', 'log2']
        }
        logging.info("✅ Using Random Forest with parameter grid")
        
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 5, 7],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
            'model__subsample': [0.8, 0.9, 1.0]
        }
        logging.info("✅ Using Gradient Boosting with parameter grid")
        
    elif model_type == "ridge":
        model = Ridge(random_state=42)
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
        logging.info("✅ Using Ridge with parameter grid")
        
    elif model_type == "lasso":
        model = Lasso(random_state=42)
        param_grid = {
            'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        }
        logging.info("✅ Using Lasso with parameter grid")
        
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    # --- Perform Hyperparameter Search ---
    logging.info("🔍 Starting hyperparameter search...")
    
    if search_type == "grid":
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=10,
            cv=cv_folds,
            scoring='r2',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
    else:
        raise ValueError(f"Unsupported search_type: {search_type}")
    
    # Start MLflow run
    mlflow.start_run(run_name="hyperparameter_tuning", nested=True)
    
    try:
        # Fit the search
        logging.info(f"Training {search_type.upper()}Search with {cv_folds} CV folds...")
        search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = search.best_score_
        
        logging.info(f"✅ Best R² Score (CV): {best_score:.4f}")
        logging.info(f"✅ Best Parameters: {best_params}")
        
        # Log to MLflow
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_r2", best_score)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("search_type", search_type)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.sklearn.log_model(best_model, "best_model")
        
        return best_model, best_params
        
    except Exception as e:
        logging.error(f"❌ Error during hyperparameter tuning: {e}")
        raise e
        
    finally:
        mlflow.end_run()