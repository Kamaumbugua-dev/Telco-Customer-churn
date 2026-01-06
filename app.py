# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


# --- 1. Load model and feature names ---
ARTIFACTS_DIR = r"C:\Users\Hp\churn_app\artifacts"
model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
features_path = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")

model = joblib.load(model_path)
feature_names = joblib.load(features_path)

# --- 2. Load original dataset for feature options ---
DATA_PATH = r"C:\Users\Hp\churn_app\data\churn.csv"
df = pd.read_csv(DATA_PATH)

# --- 3. Separate categorical and numeric features ---
TARGET_COL = "Actual_Churn"
DROP_COLS = ["Actual_Churn", "Predicted_Churn", "Churn_Probability", "Risk_Segment"]
X = df.drop(columns=DROP_COLS)

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

st.title("Customer Churn Prediction")

st.sidebar.header("Input Customer Data")
input_data = {}

# --- 4. Create sliders for numeric columns ---
for col in num_cols:
    min_val = float(X[col].min())
    max_val = float(X[col].max())
    mean_val = float(X[col].mean())
    input_data[col] = st.sidebar.slider(col, min_value=min_val, max_value=max_val, value=mean_val)

# --- 5. Create dropdowns for categorical columns ---
for col in cat_cols:
    options = X[col].dropna().unique().tolist()
    input_data[col] = st.sidebar.selectbox(col, options)

# --- 6. Predict on button click ---
if st.sidebar.button("Predict Churn"):
    input_df = pd.DataFrame([input_data])
    
    # --- 7. Predict ---
    pred_proba = model.predict_proba(input_df)[:, 1][0]
    pred_class = model.predict(input_df)[0]
    
    # --- 8. Map risk segment ---
    risk = "High" if pred_proba >= 0.7 else "Medium" if pred_proba >= 0.4 else "Low"
    
    # --- 9. Display results ---
    st.subheader("Prediction Results")
    st.write(f"**Predicted Churn:** {pred_class}")
    st.write(f"**Churn Probability:** {pred_proba:.2f}")
    st.write(f"**Risk Segment:** {risk}")
    
    # --- 10. Smart translation with reasons ---
    reasons = []
    suggestions = []

    # Example reasons based on features
    if "Contract" in input_data and input_data["Contract"] == "Month-to-month":
        reasons.append("The customer is on a month-to-month contract, which is more likely to churn.")
        suggestions.append("Consider offering long-term contracts or loyalty benefits.")

    if "tenure" in input_data and input_data["tenure"] < 12:
        reasons.append(f"The customer has only been with the company for {input_data['tenure']} months, which increases churn risk.")
        suggestions.append("Engage new customers with welcome offers and onboarding support.")

    if "OnlineSecurity" in input_data and input_data["OnlineSecurity"] == "No":
        reasons.append("The customer does not have online security service, which can reduce satisfaction.")
        suggestions.append("Offer online security packages to improve retention.")

    if "PaymentMethod" in input_data and input_data["PaymentMethod"] == "Electronic check":
        reasons.append("Electronic check payment method has a higher churn tendency.")
        suggestions.append("Encourage more reliable payment options like auto-pay.")

    if "MonthlyCharges" in input_data and input_data["MonthlyCharges"] > df["MonthlyCharges"].mean():
        reasons.append(f"Monthly charges ({input_data['MonthlyCharges']}) are above average, which can lead to dissatisfaction.")
        suggestions.append("Offer discounts or bundle packages to reduce cost concerns.")

    # Default if no specific reason
    if not reasons:
        reasons.append("No high-risk flags found based on the selected parameters.")
        suggestions.append("Keep monitoring customer satisfaction.")

    # --- 11. Show translation and suggestions ---
    st.subheader(" Translation with Reasons")
    st.write("- Customer Churn Prediction → Forecast of whether a customer will leave.")
    st.write("- Prediction Results → The outcome of the forecast.")
    st.write(f"- Predicted Churn: {pred_class} → {'The model predicts this customer will not leave (0 = no churn).' if pred_class==0 else 'The model predicts this customer may leave (1 = churn).'}")
    st.write(f"- Churn Probability: {pred_proba:.2f} → There’s a {pred_proba*100:.0f}% chance the customer might leave.")
    st.write(f"- Risk Segment: {risk} → The customer is considered {risk.lower()} risk of leaving.")

    st.subheader(" Reasons Based on Input Parameters")
    for r in reasons:
        st.write(f"- {r}")

    st.subheader(" Suggestions to Improve Retention")
    for s in suggestions:
        st.write(f"- {s}")

    # --- 12. Plain words explanation ---
    plain_words = (
        f" In plain words:\n"
        f"The system thinks this customer is likely to {'stay' if pred_class==0 else 'leave'}, "
        f"with a {pred_proba*100:.0f}% chance of leaving, "
        f"so they’re classified as {risk.lower()} risk.\n"
        f"Reasons: {', '.join([r for r in reasons])}\n"
        f"Suggested actions: {', '.join([s for s in suggestions])}"
    )
    st.info(plain_words)
