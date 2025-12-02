import pandas as pd

# Load your statutes file
df = pd.read_csv(r"C:\Users\hitik\OneDrive\Desktop\legaltalk\LegalTalk\data\statutes.csv")

print("### Shape of dataset ###")
print(df.shape)

print("\n### Columns ###")
print(df.columns.tolist())

print("\n### Data Types ###")
print(df.dtypes)

print("\n### Missing Values Per Column ###")
print(df.isna().sum())

print("\n### Missing Value Percentage ###")
print((df.isna().mean() * 100).round(2))

print("\n### Sample Rows ###")
print(df.head(10))

print("\n### Unique Statute IDs Count ###")
print(df['id'].nunique())

print("\n### Sample IDs ###")
print(df['id'].unique()[:20])