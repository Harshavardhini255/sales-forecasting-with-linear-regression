"""Streamlit app: predict monthly sales from a marketing budget mix.

Runs the best regression model saved by model_comparison.py.
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="centered")

st.title("📈 Sales Forecast Playground")
st.markdown(
    "Drag the sliders to set next month's marketing budget and see the "
    "predicted sales. The model is a **multiple linear regression** trained on "
    "96 months of data, selected by cross-validation."
)

@st.cache_resource
def load_model():
    bundle = joblib.load("output/sales_model.joblib")
    return bundle["model"], bundle["features"]


model, features = load_model()

st.subheader("Marketing budget")

instagram = st.slider("Instagram ad spend ($)", 0, 10000, 2500, 100)
tv = st.slider("TV ad spend ($)", 0, 50000, 12000, 500)
email = st.slider("Email spend ($)", 0, 10000, 2000, 100)
season = st.selectbox("Season", ["Q1", "Q2", "Q3", "Q4"], index=0)

row = {f: 0 for f in features}
row["instagram_ad_spend"] = instagram
row["tv_ad_spend"] = tv
row["email_spend"] = email
for quarter in ["Q2", "Q3", "Q4"]:
    row[f"season_{quarter}"] = int(season == quarter)

X_input = pd.DataFrame([row])[features]
prediction = float(model.predict(X_input)[0])

st.divider()
st.metric("Predicted monthly sales", f"${prediction:,.0f}")

spend = instagram + tv + email
if spend > 0:
    st.caption(f"Return: ${prediction / spend:,.2f} per $1 of total budget")

st.caption("Bundle: sales_model.joblib (Multiple Linear Regression, chosen by 5-fold CV)")