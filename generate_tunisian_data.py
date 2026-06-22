import pandas as pd
import numpy as np
import random
from datetime import datetime

# Define Tunisian cities with their characteristics
tunisian_cities = {
    "Tunis": {"base_price": 350000, "price_variation": 150000},
    "Sfax": {"base_price": 250000, "price_variation": 100000},
    "Sousse": {"base_price": 300000, "price_variation": 120000},
    "Monastir": {"base_price": 280000, "price_variation": 110000},
    "Nabeul": {"base_price": 220000, "price_variation": 90000},
    "Hammamet": {"base_price": 400000, "price_variation": 180000},
    "Bizerte": {"base_price": 200000, "price_variation": 80000},
    "Gabes": {"base_price": 150000, "price_variation": 60000},
    "Kairouan": {"base_price": 180000, "price_variation": 70000},
    "Mahdia": {"base_price": 210000, "price_variation": 85000},
    "Sidi Bou Said": {"base_price": 450000, "price_variation": 200000},
    "La Marsa": {"base_price": 420000, "price_variation": 190000},
    "Gammarth": {"base_price": 500000, "price_variation": 250000},
    "Djerba": {"base_price": 380000, "price_variation": 160000},
    "Tozeur": {"base_price": 160000, "price_variation": 65000}
}

# Define Tunisian neighborhoods types
neighborhood_types = [
    "Centre Ville", "Résidentiel", "Luxueux", "Populaire", 
    "Périphérie", "Touristique", "Agricole", "Côtière"
]

# Define building types common in Tunisia
building_types = [
    "Appartement", "Maison", "Villa", "Duplex", "Studio", 
    "Penthouse", "Riad", "Dar (Maison Traditionnelle)"
]

def generate_tunisian_housing_data(n_samples=2000):
    """
    Generate synthetic Tunisian housing data
    """
    np.random.seed(42)
    random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Select random city
        city = random.choice(list(tunisian_cities.keys()))
        city_data = tunisian_cities[city]
        
        # Generate features
        surface_area = np.random.normal(120, 50)  # m²
        surface_area = max(20, min(400, surface_area))
        
        # Number of rooms
        rooms = max(1, int(np.random.normal(3.5, 1.5)))
        
        # Number of bedrooms (usually rooms - living room)
        bedrooms = max(1, int(np.random.normal(rooms - 1, 0.5)))
        
        # Number of bathrooms
        bathrooms = max(1, int(np.random.normal(bedrooms * 0.6, 0.3)))
        
        # Year built
        current_year = datetime.now().year
        year_built = np.random.randint(1960, current_year + 1)
        age = current_year - year_built
        
        # Floor level
        floor_level = np.random.randint(0, 6)
        has_elevator = random.choice([True, False]) if floor_level > 2 else False
        
        # Amenities
        has_parking = random.choice([True, False])
        has_terrace = random.choice([True, False])
        has_garden = random.choice([True, False])
        has_pool = random.choice([True, False]) if random.random() > 0.8 else False
        
        # Neighborhood type
        neighborhood = random.choice(neighborhood_types)
        
        # Building type
        building_type = random.choice(building_types)
        
        # Price calculation based on multiple factors
        base_price = city_data["base_price"]
        
        # Factor adjustments
        price_multiplier = 1.0
        
        # Surface area effect
        price_multiplier += (surface_area - 100) * 0.005
        
        # Rooms effect
        price_multiplier += (rooms - 3) * 0.05
        
        # Age effect (older = less value)
        age_factor = 1 - (age / 100) * 0.3
        price_multiplier *= age_factor
        
        # Neighborhood effect
        neighborhood_factors = {
            "Centre Ville": 1.2,
            "Résidentiel": 1.1,
            "Luxueux": 1.4,
            "Populaire": 0.8,
            "Périphérie": 0.9,
            "Touristique": 1.3,
            "Agricole": 0.7,
            "Côtière": 1.25
        }
        price_multiplier *= neighborhood_factors.get(neighborhood, 1.0)
        
        # Building type effect
        building_factors = {
            "Appartement": 1.0,
            "Maison": 1.2,
            "Villa": 1.5,
            "Duplex": 1.3,
            "Studio": 0.7,
            "Penthouse": 1.4,
            "Riad": 1.6,
            "Dar (Maison Traditionnelle)": 1.2
        }
        price_multiplier *= building_factors.get(building_type, 1.0)
        
        # Amenities effects
        if has_parking: price_multiplier *= 1.1
        if has_terrace: price_multiplier *= 1.05
        if has_garden: price_multiplier *= 1.15
        if has_pool: price_multiplier *= 1.2
        if has_elevator: price_multiplier *= 1.1
        
        # Random noise
        price_multiplier *= np.random.normal(1.0, 0.1)
        
        # Calculate final price
        price = base_price * price_multiplier
        price = max(50000, min(2000000, int(price / 1000) * 1000))  # Round to nearest 1000 TND
        
        data.append({
            'city': city,
            'neighborhood': neighborhood,
            'address': f"{neighborhood}, {city}",
            'building_type': building_type,
            'surface_area_m2': surface_area,
            'rooms': rooms,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'floor_level': floor_level,
            'has_elevator': has_elevator,
            'year_built': year_built,
            'age': age,
            'has_parking': has_parking,
            'has_terrace': has_terrace,
            'has_garden': has_garden,
            'has_pool': has_pool,
            'price_tnd': price
        })
    
    return pd.DataFrame(data)

# Generate the dataset
print("🏠 Generating Tunisian housing dataset...")
df = generate_tunisian_housing_data(2000)

# Create data folder if it doesn't exist
import os
if not os.path.exists('data'):
    os.makedirs('data')

# Save to CSV
df.to_csv('data/tunisian_housing_data.csv', index=False)
print(f"✅ Dataset generated successfully!")
print(f"📊 Shape: {df.shape}")
print(f"📊 Columns: {df.columns.tolist()}")
print(f"\n📋 Sample data:")
print(df.head())
print(f"\n📊 Price Statistics:")
print(df['price_tnd'].describe())
print(f"\n📊 City Distribution:")
print(df['city'].value_counts().head())