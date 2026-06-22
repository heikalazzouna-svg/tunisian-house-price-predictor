import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

print("🏠 Testing Tunisian Data Pipeline Readiness")
print("=" * 50)

# Load the prepared data
df = pd.read_csv('data/prepared_tunisian_data.csv')
print(f"✅ Data loaded: {df.shape}")

# Check columns
print(f"\n📊 Columns: {df.columns.tolist()}")

# Split features and target
X = df.drop(columns=['price_tnd', 'price_per_m2'])
y = df['price_tnd']

print(f"\n📊 Features: {X.shape[1]}")
print(f"📊 Target: {y.name}")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train simple model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Model Performance:")
print(f"✅ R² Score: {r2:.4f}")
print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Test samples: {len(X_test)}")

print("\n✅ Data is ready for ZenML pipeline!")
print("Run: python run_tunisian_pipeline.py")