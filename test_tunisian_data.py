import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def test_tunisian_data():
    """Test the Tunisian data with a simple model"""
    
    # Load the prepared data
    try:
        X = pd.read_csv('data/X_features.csv')
        y = pd.read_csv('data/y_target.csv')
        
        print("✅ Data loaded successfully!")
        print(f"📊 X shape: {X.shape}")
        print(f"📊 y shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train a simple model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        print(f"\n📊 Model Performance on Tunisian Data:")
        print(f"✅ R² Score: {r2:.4f}")
        print(f"✅ MSE: {mse:.2f}")
        
        # Show price predictions
        print(f"\n📊 Sample Predictions vs Actual:")
        # Convert to 1D array if needed
        y_test_flat = y_test.values.flatten() if hasattr(y_test, 'values') else y_test
        for i in range(min(5, len(y_pred))):
            pred_val = y_pred[i] if not isinstance(y_pred[i], (list, np.ndarray)) else y_pred[i][0]
            actual_val = y_test_flat[i] if not isinstance(y_test_flat[i], (list, np.ndarray)) else y_test_flat[i][0]
            print(f"   Predicted: {pred_val:.2f} TND | Actual: {actual_val:.2f} TND")
        
        # Feature importance
        print(f"\n📊 Top 5 Most Important Features:")
        importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        })
        importance['abs_coef'] = importance['coefficient'].abs()
        importance = importance.sort_values('abs_coef', ascending=False).head(5)
        for _, row in importance.iterrows():
            print(f"   {row['feature']}: {row['coefficient']:.2f}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Please run generate_tunisian_data.py and adapt_to_tunisian_data.py first.")
        print("   Then try: python test_tunisian_data.py")

if __name__ == "__main__":
    test_tunisian_data()