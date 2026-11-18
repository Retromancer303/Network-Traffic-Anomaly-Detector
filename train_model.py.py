
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/traffic.csv", low_memory=False)
df.columns = df.columns.str.strip()

# Convert labels into binary categories
df["Label"] = (df["Label"].astype(str).str.strip() != "BENIGN").astype(int)

# Choose a few initial network features
features = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Flow Bytes/s"
]

# Keep only the columns we need
df = df[features + ["Label"]].copy()

# Convert feature values into numbers
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

# Remove invalid values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Separate inputs and expected outputs
X = df[features]
y = df["Label"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Display results
print(classification_report(
    y_test,
    predictions,
    target_names=["Normal", "Attack"],
    zero_division=0
))

# Save the trained model and feature names
joblib.dump(
    {"model": model, "features": features},
    "models/network_model.pkl"
)

print("Model saved successfully!")