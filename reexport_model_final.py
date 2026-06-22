import mlflow
import joblib
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("🏠 Re-exporting Model with Scikit-learn 1.5.2")
print("=" * 60)
print(f"Scikit-learn version: {sklearn.__version__}")

# Your Run ID
RUN_ID = '9760635d468a4b4ca7b2c1417a3fc75b'

# Set tracking URI
mlflow.set_tracking_uri('sqlite:///C:/Users/LENOVO/AppData/Roaming/zenml/local_stores/66893170-5ab4-4e6f-9db1-a846999df821/mlflow.db')

try:
    # Load the model from MLflow
    model = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/model')
    print("✅ Model loaded from MLflow")
    
    # Extract the actual pipeline from the loaded model
    if hasattr(model, '_model_impl'):
        # MLflow wrapper
        pipeline = model._model_impl
    elif hasattr(model, 'named_steps'):
        # Direct pipeline
        pipeline = model
    else:
        pipeline = model
    
    print(f"Pipeline type: {type(pipeline)}")
    
    # Save as pickle with protocol 4 (most compatible)
    with open('model.pkl', 'wb') as f:
        pickle.dump(pipeline, f, protocol=4)
    print("✅ Model saved as pickle (protocol 4)")
    
    # Also save as joblib
    joblib.dump(pipeline, 'model_joblib.pkl', compress=3)
    print("✅ Model saved as joblib")
    
    # TEST: Try loading and predicting
    print("\n🔍 Testing the saved model...")
    with open('model.pkl', 'rb') as f:
        test_model = pickle.load(f)
    
    # Create sample input
    sample_data = pd.DataFrame([[
        120, 4, 3, 2, 1, 300000, 'Tunis', 0, 2500, 0.025,
        4.79, 1.61, 1.39, 1.10, 12.61, 0.025, 2.5
    ]], columns=['surface_area_m2', 'total_rooms', 'rooms', 'bathrooms', 
                 'floor_level', 'prix', 'city', 'city_encoded', 'price_per_m2',
                 'rooms_per_surface', 'surface_area_m2_log', 'total_rooms_log',
                 'rooms_log', 'bathrooms_log', 'prix_log', 'rooms_per_surface_log',
                 'rooms_per_100sqm'])
    
    prediction = test_model.predict(sample_data)
    print(f"✅ Test prediction: {prediction[0]:,.0f} TND")
    print("\n✅ Model is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")