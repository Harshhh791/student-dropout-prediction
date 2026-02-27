import pandas as pd
import sys

print("Python:", sys.executable)

df = pd.read_csv("data/data.csv", sep=";")
print(df["Target"].value_counts())
print(df.isna().sum().sort_values(ascending=False).head(10))
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:10])
print(df.head(3))
