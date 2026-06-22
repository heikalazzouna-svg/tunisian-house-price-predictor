import mlflow
import pandas as pd
import joblib
import os
from zenml.client import Client
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

print("🏠 Exporting Best Model from MLflow")
print("=" * 50)

# Set MLflow tracking URI
try:
    tracking_uri = get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✅ MLflow tracking URI: {tracking_uri}")
except:
    mlflow.set_tracking_uri("file:./mlruns")
    print("⚠️ Using local MLflow")

# Get the latest run from MLflow
client = Client()
runs = client.list_pipeline_runs()

# Find the latest successful run with the best R²
best_run = None
best_r2 = -1

for run in runs:
    if run.status == "COMPLETED":
        # Get metrics from MLflow
        try:
            experiment = mlflow.get_experiment(run.experiment_id)
            runs_mlflow = mlflow.search_runs(
                experiment_ids=[run.experiment_id],
                filter_string=f"tags.run_id = '{run.id}'"
            )
            if not runs_mlflow.empty:
                # Look for R² score
                r2_cols = [col for col in runs_mlflow.columns if 'r2' in col.lower() or 'score' in col.lower()]
                for col in r2_cols:
                    if pd.notna(runs_mlflow[col].values[0]):
                        r2 = float(runs_mlflow[col].values[0])
                        if r2 > best_r2:
                            best_r2 = r2
                            best_run = run
                            print(f"✅ Found run with R²: {r2:.4f}")
                            break
        except:
            pass

if best_run is None:
    print("❌ No runs found with metrics. Using the latest run.")
    # Fallback to latest run
    best_run = runs[0] if runs else None

if best_run is None:
    print("❌ No pipeline runs found!")
    print("Please run the pipeline first: python run_real_tunisian_pipeline.py")
    exit(1)

print(f"\n📊 Best Run:")
print(f"   Pipeline: {best_run.pipeline.name}")
print(f"   Run ID: {best_run.id}")
print(f"   R² Score: {best_r2:.4f}")

# Load the model from MLflow
try:
    # Get the model artifact
    from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    
    # Find the model server
    services = deployer.find_model_server(
        pipeline_name=best_run.pipeline.name,
        pipeline_step_name="model_building_step"
    )
    
    if services:
        model_uri = services[0].model_uri
        print(f"✅ Found model URI: {model_uri}")
        model = mlflow.sklearn.load_model(model_uri)
    else:
        # Alternative: try to load from artifacts
        artifact_uri = f"runs:/{best_run.id}/model"
        print(f"⚠️ Using artifact URI: {artifact_uri}")
        model = mlflow.sklearn.load_model(artifact_uri)
    
    # Save the model
    joblib.dump(model, 'model.pkl')
    print(f"✅ Model saved to: model.pkl")
    print(f"📊 Model type: {type(model)}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\n🔄 Trying alternative method...")
    
    try:
        # Alternative: try to load from MLflow directly
        experiment = mlflow.get_experiment(best_run.experiment_id)
        runs_mlflow = mlflow.search_runs(experiment_ids=[best_run.experiment_id])
        
        if not runs_mlflow.empty:
            run_id = runs_mlflow.iloc[0].run_id
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            joblib.dump(model, 'model.pkl')
            print(f"✅ Model saved to: model.pkl")
        else:
            print("❌ Could not find model in MLflow")
    except Exception as e2:
        print(f"❌ Alternative method also failed: {e2}")