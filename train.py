# train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# --- 1. Load the data ---
DATA_PATH = r"C:\Users\Hp\churn_app\data\churn.csv"
df = pd.read_csv(DATA_PATH)

# --- 2. Define target and features ---
TARGET_COL = "Actual_Churn"

# Drop columns that are not features
DROP_COLS = ["Actual_Churn", "Predicted_Churn", "Churn_Probability", "Risk_Segment"]
X = df.drop(columns=DROP_COLS)
y = df[TARGET_COL]

# Remove rows where target is NaN
mask = y.notna()
X = X[mask]
y = y[mask]

if len(y) == 0:
    raise ValueError("No valid target rows found. Check your CSV data for missing values in Actual_Churn.")

# --- 3. Identify categorical and numerical columns ---
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# --- 4. Build preprocessing pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ],
    remainder="passthrough"  # Keep numeric columns as is
)

# --- 5. Build model pipeline ---
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- 6. Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 7. Train the model ---
model_pipeline.fit(X_train, y_train)

# --- 8. Save artifacts ---
ARTIFACTS_DIR = r"C:\Users\Hp\churn_app\artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

joblib.dump(model_pipeline, os.path.join(ARTIFACTS_DIR, "model.pkl"))

# Save feature names for Streamlit display
feature_names = model_pipeline.named_steps["preprocessor"].get_feature_names_out()
joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, "feature_names.pkl"))

print("Training complete. Model and features saved to artifacts/")
