import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Churn Prediction System", layout="wide")
st.title("🔁 Customer Churn – Multi Model Train & Predict System")

os.makedirs("saved_models", exist_ok=True)

# =========================
# SIDEBAR
# =========================
mode = st.sidebar.radio("Select Mode", ["Train Model", "Predict Churn"])

# =========================
# TRAIN MODE
# =========================
if mode == "Train Model":

    st.subheader("📊 Train Models (Dataset WITH Churn)")

    file = st.file_uploader("Upload Dataset WITH Churn Column", type=["csv"])
    if file is None:
        st.info("ℹ️ Please upload a dataset to continue.")
        st.stop()

    df = pd.read_csv(file)
    st.dataframe(df.head(), use_container_width=True)

    target = st.selectbox(
        "Select Churn Column",
        ["-- Select Churn Column --"] + df.columns.tolist()
    )

    if target == "-- Select Churn Column --":
        st.stop()

    if df[target].nunique() > 5:
        st.warning("⚠️ Churn column should be binary (0/1).")
        st.stop()

    df = df.drop(columns=["customer_id"], errors="ignore")

    y = df[target]
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)

    if y.value_counts().min() >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        st.warning("⚠️ Highly imbalanced churn data. Stratify skipped.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1 Score": f1_score(y_test, preds, zero_division=0),
            "model": model
        }

    results_df = pd.DataFrame(results).T.drop("model", axis=1).fillna(0)

    st.subheader("📈 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    fig, ax = plt.subplots()
    results_df.plot(kind="bar", ax=ax)
    plt.xticks(rotation=0)
    st.pyplot(fig)

    # ---------- Non-Tech Score ----------
    st.subheader("🏆 Model Score (Out of 100)")

    score_df = pd.DataFrame({
        "Model": results_df.index,
        "Score (Out of 100)": (results_df["F1 Score"] * 100).round(0).astype(int)
    })

    st.dataframe(score_df, use_container_width=True)

    best_model = score_df.sort_values("Score (Out of 100)", ascending=False).iloc[0]

    st.success(
        f"""
✅ **Best Model: {best_model['Model']}**

📊 Score: **{best_model['Score (Out of 100)']} / 100**

👉 Recommended for this dataset.
"""
    )

    model_to_save = st.selectbox("Select Model to SAVE", score_df["Model"])

    if st.button("Save Selected Model"):
        joblib.dump(results[model_to_save]["model"], f"saved_models/{model_to_save}.pkl")
        joblib.dump(scaler, "saved_models/scaler.pkl")
        joblib.dump(X.columns.tolist(), "saved_models/features.pkl")
        st.success("✅ Model saved")

    st.stop()

# =========================
# PREDICT MODE
# =========================
if mode == "Predict Churn":

    st.subheader("🤖 Predict Churn (Dataset WITHOUT Churn)")

    file = st.file_uploader("Upload Dataset WITHOUT Churn Column", type=["csv"])
    if file is None:
        st.stop()

    model_files = [
        f for f in os.listdir("saved_models")
        if f.endswith(".pkl") and f not in ["scaler.pkl", "features.pkl"]
    ]

    model_name = st.selectbox("Select Saved Model", model_files)

    model = joblib.load(f"saved_models/{model_name}")
    scaler = joblib.load("saved_models/scaler.pkl")
    features = joblib.load("saved_models/features.pkl")

    df = pd.read_csv(file)
    st.dataframe(df.head(), use_container_width=True)

    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=features, fill_value=0)

    X_scaled = scaler.transform(df_encoded)

    df["Predicted_Churn"] = model.predict(X_scaled)
    df["Churn_Probability (%)"] = (model.predict_proba(X_scaled)[:, 1] * 100).round(2)

    def risk_action(p):
        if p >= 80:
            return "High Risk", "📞 Call Immediately"
        elif p >= 50:
            return "Medium Risk", "🎁 Offer Discount"
        else:
            return "Low Risk", "✅ Ignore"

    df[["Risk_Level", "Action"]] = df["Churn_Probability (%)"].apply(
        lambda x: pd.Series(risk_action(x))
    )

    st.subheader("📄 Prediction Result")
    st.dataframe(df.head(), use_container_width=True)

    # ---------- Graph ----------
    st.subheader("📊 Churn Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Churn_Probability (%)"], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # ---------- Probability Table ----------
    prob_table = (
        pd.cut(df["Churn_Probability (%)"], bins=[0,50,80,100])
        .value_counts()
        .reset_index()
    )
    prob_table.columns = ["Probability Range", "Customers"]
    prob_table["% Customers"] = (prob_table["Customers"] / len(df) * 100).round(2)

    st.subheader("📋 Churn Probability Summary")
    st.dataframe(prob_table)

    # ---------- Risk Graph ----------
    st.subheader("🚦 Risk Level Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Risk_Level", data=df, ax=ax)
    st.pyplot(fig)

    # ---------- Risk Table ----------
    risk_table = df["Risk_Level"].value_counts().reset_index()
    risk_table.columns = ["Risk Level", "Customers"]
    risk_table["% Customers"] = (risk_table["Customers"] / len(df) * 100).round(2)

    st.subheader("📋 Risk Level Summary")
    st.dataframe(risk_table)

    # ---------- Business Insight ----------
    high_risk_pct = risk_table.loc[
        risk_table["Risk Level"] == "High Risk", "% Customers"
    ].values[0]

    st.info(
        f"📌 **Insight:** {high_risk_pct}% customers are at HIGH risk of churn. "
        f"Immediate retention action is recommended."
    )

    # ---------- CALL LIST ----------
    st.subheader("📞 Call List – High Risk Customers Only")

    high_risk_df = df[df["Risk_Level"] == "High Risk"]

    if high_risk_df.empty:
        st.success("✅ No high-risk customers found.")
    else:
        st.dataframe(high_risk_df, use_container_width=True)
        st.download_button(
            "⬇️ Download High-Risk Call List CSV",
            high_risk_df.to_csv(index=False),
            "high_risk_call_list.csv",
            "text/csv"
        )

    # ---------- Download Full CSV ----------
    st.download_button(
        "⬇️ Download Full Prediction CSV",
        df.to_csv(index=False),
        "churn_predictions.csv",
        "text/csv"
    )
