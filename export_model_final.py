import mlflow
import joblib

print("🏠 Exporting Model from MLflow")
print("=" * 50)

# Your Run ID
RUN_ID = '9760635d468a4b4ca7b2c1417a3fc75b'

# Set the tracking URI
mlflow.set_tracking_uri('sqlite:///C:/Users/LENOVO/AppData/Roaming/zenml/local_stores/66893170-5ab4-4e6f-9db1-a846999df821/mlflow.db')

print(f"🔍 Loading model from Run ID: {RUN_ID}")

try:
    # Load the model
    model = mlflow.sklearn.load_model(f'runs:/{RUN_ID}/model')
    print("✅ Model loaded successfully!")
    
    # Save the model
    joblib.dump(model, 'model.pkl')
    print("✅ Model saved to: model.pkl")
    print(f"📊 Model type: {type(model)}")
    
except Exception as e:
    print(f"❌ Error: {e}")