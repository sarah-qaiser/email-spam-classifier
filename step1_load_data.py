#Data Loading & Exploration
import pandas as pd
# 1. Load the dataset
df = pd.read_csv("C:\\Users\\THIS PC\\Downloads\\spam.csv", encoding='latin-1')

# 2. Drop unnamed/empty columns
df = df[['v1', 'v2']]

# 3. Rename columns to something meaningful
df.columns = ['label', 'text']

# 4. Basic exploration 
print("=== Shape ===")
print(df.shape)

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Label Distribution ===")
print(df['label'].value_counts())

print("\n=== Missing Values ===")
print(df.isnull().sum())