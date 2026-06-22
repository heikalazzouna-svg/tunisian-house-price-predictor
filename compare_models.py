import pandas as pd
from zenml.client import Client
import mlflow

print("🏠 Model Performance Comparison")
print("=" * 50)

# Get runs from MLflow
client = Client()
runs = client.list_pipeline_runs()

for run in runs:
    print(f"\n📊 Run: {run.name}")
    print(f"   Pipeline: {run.pipeline.name}")
    print(f"   Status: {run.status}")
    if run.status == "COMPLETED":
        # Get metrics from MLflow
        print(f"   Check MLflow UI for metrics")