import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
import sys
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🏠 Tunisian House Price Predictor", layout="wide")

st.title("🏠 Tunisian House Price Predictor")
st.markdown("### Predict house prices in Tunisia using Machine Learning")

# --- REAL FIX: Monkey patch scikit-learn to handle missing attribute ---
def fix_sklearn_compatibility():
    """Fix scikit-learn compatibility issue with monotonic_cst"""
    try:
        import sklearn
        import sklearn.tree
        
        # Store original DecisionTreeRegressor
        original_dtr = sklearn.tree.DecisionTreeRegressor
        
        # Create a wrapper class that handles missing attributes
        class FixedDecisionTreeRegressor(original_dtr):
            def __init__(self, *args, **kwargs):
                # Remove monotonic_cst if present (it doesn't exist in older versions)
                if 'monotonic_cst' in kwargs:
                    kwargs.pop('monotonic_cst')
                super().__init__(*args, **kwargs)
            
            def __getattribute__(self, name):
                # Intercept attempts to access monotonic_cst
                if name == 'monotonic_cst':
                    return None
                return super().__getattribute__(name)
        
        # Replace the class
        sklearn.tree.DecisionTreeRegressor = FixedDecisionTreeRegressor
        return True
    except Exception as e:
        print(f"Compatibility fix failed: {e}")
        return False

# Apply the fix BEFORE loading the model
fix_sklearn_compatibility()

# Load model
@st.cache_resource
def load_model():
    try:
        # Try loading with pickle
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ Model loaded with pickle!")
        return model
    except Exception as e:
        st.error(f"❌ Model not found: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# --- SIDEBAR INPUTS ---
st.sidebar.header("📊 Property Features")
st.sidebar.markdown("---")

# User inputs
surface = st.sidebar.number_input("Surface Area (m²)", min_value=20, max_value=500, value=120, step=5)
rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=10, value=3, step=1)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2, step=1)
floor = st.sidebar.number_input("Floor Level", min_value=0, max_value=10, value=1, step=1)

# City selection with proper encoding
city_options = {
    "Tunis": 0,
    "Sfax": 1,
    "Sousse": 2,
    "Hammamet": 3,
    "Monastir": 4,
    "Nabeul": 5,
    "Bizerte": 6,
    "Gabes": 7,
    "Kairouan": 8,
    "Mahdia": 9,
    "Sidi Bou Said": 10,
    "La Marsa": 11,
    "Gammarth": 12,
    "Djerba": 13,
    "Tozeur": 14
}

city_name = st.sidebar.selectbox("City", list(city_options.keys()))
city_encoded = city_options[city_name]

# --- CALCULATE ENGINEERED FEATURES ---
total_rooms = rooms + 1

# Price per m2 based on city
city_prices = {
    "Tunis": 3500, "Sfax": 2500, "Sousse": 2800, 
    "Hammamet": 4000, "Monastir": 3000, "Nabeul": 2200, 
    "Bizerte": 2000, "Gabes": 1500, "Kairouan": 1800,
    "Mahdia": 2100, "Sidi Bou Said": 4500, "La Marsa": 4200,
    "Gammarth": 5000, "Djerba": 3800, "Tozeur": 1600
}
price_per_m2 = city_prices.get(city_name, 2500)

# Derived features
surface_area_m2 = float(surface)
rooms_per_surface = rooms / surface if surface > 0 else 0
surface_area_m2_log = np.log1p(surface)
total_rooms_log = np.log1p(total_rooms)
rooms_log = np.log1p(rooms)
bathrooms_log = np.log1p(bathrooms)

prix = price_per_m2 * surface
prix_log = np.log1p(prix)
rooms_per_surface_log = np.log1p(rooms_per_surface)
rooms_per_100sqm = rooms / (surface / 100) if surface > 0 else 0

# --- PREPARE INPUT DATA ---
input_data = pd.DataFrame([[
    surface_area_m2,           # surface_area_m2
    total_rooms,               # total_rooms
    rooms,                     # rooms
    bathrooms,                 # bathrooms
    floor,                     # floor_level
    prix,                      # prix
    city_name,                 # city
    city_encoded,              # city_encoded
    price_per_m2,              # price_per_m2
    rooms_per_surface,         # rooms_per_surface
    surface_area_m2_log,       # surface_area_m2_log
    total_rooms_log,           # total_rooms_log
    rooms_log,                 # rooms_log
    bathrooms_log,             # bathrooms_log
    prix_log,                  # prix_log
    rooms_per_surface_log,     # rooms_per_surface_log
    rooms_per_100sqm           # rooms_per_100sqm
]], columns=[
    'surface_area_m2', 'total_rooms', 'rooms', 'bathrooms', 'floor_level',
    'prix', 'city', 'city_encoded', 'price_per_m2',
    'rooms_per_surface', 'surface_area_m2_log', 'total_rooms_log',
    'rooms_log', 'bathrooms_log', 'prix_log', 'rooms_per_surface_log',
    'rooms_per_100sqm'
])

# --- PREDICT ---
if st.sidebar.button("🔮 Predict Price", type="primary"):
    try:
        with st.spinner("Calculating price..."):
            # Use the model's predict method directly
            prediction = model.predict(input_data)[0]
            
            st.markdown("---")
            st.markdown("## 🏠 Price Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.metric(
                    label="Predicted House Price",
                    value=f"{prediction:,.0f} TND",
                    delta=f"Based on {surface} m² in {city_name}"
                )
            
            st.markdown("### 📍 Property Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Surface Area", f"{surface} m²")
            with col2:
                st.metric("Rooms", rooms)
            with col3:
                st.metric("Bathrooms", bathrooms)
            with col4:
                st.metric("Floor", floor)
            
            st.markdown("### 📊 Price Breakdown")
            st.markdown(f"""
            - **Base Price (per m²):** {price_per_m2:,.0f} TND
            - **Surface Area:** {surface} m²
            - **Estimated Price:** {prediction:,.0f} TND
            """)
            
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.info("Please try adjusting the input values.")

# --- FOOTER ---
st.markdown("---")
st.markdown("""
### 📝 About This Predictor
This model was trained on **real Tunisian housing data** from:
- 🇹🇳 Tayara.tn
- 🇹🇳 Menzili.tn  
- 🇹🇳 Mubawab.tn

**Technology:** ZenML · MLflow · Scikit-learn · Streamlit
""")