import joblib
import pickle
import mlflow
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

print("🏠 Exporting Model with Compatible Format")
print("=" * 50)

# Your Run ID
RUN_ID = '9760635d468a4b4ca7b2c1417a3fc75b'

# Set tracking URI
mlflow.set_tracking_uri('sqlite:///C:/Users/LENOVO/AppData/Roaming/zenml/local_stores/66893170-5ab4-4e6f-9db1-a846999df821/mlflow.db')

try:
    # Load the model from MLflow
    model = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/model')
    print("✅ Model loaded from MLflow")
    
    # Save as pickle (more compatible than joblib)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("✅ Model saved as pickle")
    
    # Also save as joblib (backup)
    joblib.dump(model, 'model_joblib.pkl')
    print("✅ Model saved as joblib")
    
except Exception as e:
    print(f"❌ Error: {e}")