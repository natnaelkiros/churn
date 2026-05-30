import streamlit as st
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

# Columns
num_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls",
            "Payment Delay", "Total Spend", "Last Interaction"]

cat_cols = ["Gender", "Subscription Type", "Contract Length"]

# -----------------------------
# CLEAN + LOAD TRAINING DATA
# -----------------------------
train_df = pd.read_csv("train.csv")

# Convert empty strings to NaN
train_df = train_df.replace(r'^\s*$', pd.NA, regex=True)

# Drop rows where ALL columns are NaN
train_df = train_df.dropna(how="all")

# Drop rows where ANY column is NaN
train_df = train_df.dropna()

# Select ONLY the columns used in the model
feature_cols = num_cols + cat_cols
X_train = train_df[feature_cols]
y_train = train_df["Churn"]

# -----------------------------
# PREPROCESSING PIPELINE
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# -----------------------------
# MODEL PIPELINE
# -----------------------------
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=500))
])

model.fit(X_train, y_train)

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("Customer Churn Prediction Demo")

age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure (months)", 0, 120, 12)
usage = st.number_input("Usage Frequency", 0, 50, 10)
support = st.number_input("Support Calls", 0, 50, 2)
delay = st.number_input("Payment Delay (days)", 0, 60, 5)
spend = st.number_input("Total Spend", 0.0, 5000.0, 500.0)
last_int = st.number_input("Last Interaction (days ago)", 0, 365, 10)

gender = st.selectbox("Gender", ["Male", "Female"])
sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

input_df = pd.DataFrame([{
    "Age": age,
    "Tenure": tenure,
    "Usage Frequency": usage,
    "Support Calls": support,
    "Payment Delay": delay,
    "Total Spend": spend,
    "Last Interaction": last_int,
    "Gender": gender,
    "Subscription Type": sub_type,
    "Contract Length": contract
}])

if st.button("Predict Churn"):
    prob = model.predict_proba(input_df)[0, 1]
    pred = model.predict(input_df)[0]

    st.write(f"Churn Probability: {prob:.2f}")
    st.write("Prediction: **Churn**" if pred == 1 else "Prediction: **No Churn**")
