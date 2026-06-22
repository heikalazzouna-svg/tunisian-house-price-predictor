import pandas as pd
import zipfile
import shutil

# Extract the ZIP
with zipfile.ZipFile('data/archive.zip', 'r') as z:
    z.extractall('temp_check')

# Load the data
df = pd.read_csv('temp_check/train.csv')

# Rename the column to what the code expects
df.rename(columns={'GrLivArea': 'Gr Liv Area'}, inplace=True)

# Save back
df.to_csv('temp_check/train.csv', index=False)

# Recreate the ZIP
with zipfile.ZipFile('data/archive.zip', 'w') as z:
    z.write('temp_check/train.csv', 'train.csv')

# Clean up
shutil.rmtree('temp_check')

print('✅ Column renamed successfully!')