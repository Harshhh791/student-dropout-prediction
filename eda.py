import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/data.csv", sep=";")

print("Shape:", df.shape)
print("\nTarget distribution:")
print(df["Target"].value_counts())

print("\nMissing values (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nData types:")
print(df.dtypes.value_counts())
