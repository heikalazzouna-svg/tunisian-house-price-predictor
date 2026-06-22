import pandas as pd
import os

# Load the merged dataset
df = pd.read_csv('data/real_tunisian_data/mubawab_tayara_menzili_final.csv')

print('🏠 Real Tunisian Housing Dataset')
print('=' * 50)
print(f'📊 Shape: {df.shape}')
print(f'\n📊 Columns: {df.columns.tolist()}')
print(f'\n📋 First 5 rows:')
print(df.head())
print(f'\n📊 Data types:')
print(df.dtypes)
print(f'\n📊 Missing values:')
print(df.isnull().sum())

if 'prix' in df.columns:
    print(f'\n📊 Price statistics:')
    print(df['prix'].describe())