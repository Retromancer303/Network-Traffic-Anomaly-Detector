import pandas as pd

# Load the dataset
df = pd.read_csv("data/traffic.csv", low_memory=False)

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Display basic information
print("Dataset loaded successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Display traffic categories
print(df["Label"].value_counts())
