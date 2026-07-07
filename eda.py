import pandas as pd

df = pd.read_csv("data/raw/ISIC_2019_Training_Metadata.csv")

print("--- COLONNES DISPONIBLES ---")
print(df.columns.tolist())

print("\n--- VALEURS MANQUANTES (NaN) ---")
print(df.isnull().sum())