import os

import pandas as pd
import requests
import streamlit as st

# -------------------------------------------------------------------
# Live Codespaces URL for the Flask backend on port 7860 (no trailing slash).
# Can be overridden with the BACKEND_URL environment variable, e.g.
#   docker run -e BACKEND_URL=http://backend:7860 ...
# -------------------------------------------------------------------
BACKEND_URL = os.getenv(
    "BACKEND_URL", "https://curly-enigma-jrg444vx74qc5x5x-7860.app.github.dev"
)

# The 10 input features the serialized pipeline was trained on (same names and order)
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]

st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("🛒 SuperKart Sales Prediction Dashboard")
st.caption("Predicts Product_Store_Sales_Total for a product-store combination.")
st.markdown("---")

tab1, tab2 = st.tabs(["Online Prediction", "Batch Prediction"])

# ------------------------------- ONLINE -------------------------------
with tab1:
    st.subheader("Single Product Sales Prediction")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Product details**")
        product_weight = st.number_input("Product Weight", min_value=0.0, value=12.66)
        sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input(
            "Product Allocated Area (share of display)", min_value=0.0, max_value=1.0,
            value=0.027, format="%.3f",
        )
        mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
        id_char = st.selectbox(
            "Product ID Prefix", ["FD", "DR", "NC"],
            help="FD = Food, DR = Drinks, NC = Non-consumables",
        )
        type_category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

    with col2:
        st.markdown("**Store details**")
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"], index=1)
        city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"], index=1)
        store_type = st.selectbox(
            "Store Type",
            ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"],
            index=1,
        )
        store_age = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=16, step=1)

    if st.button("Predict Sales", type="primary"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": id_char,
            "Store_Age_Years": int(store_age),
            "Product_Type_Category": type_category,
        }
        try:
            res = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=30)
            if res.status_code == 200:
                pred = res.json().get("prediction", 0)
                st.success(f"Predicted Product-Store Sales Total: **{pred:,.2f}**")
                with st.expander("Request payload sent to API"):
                    st.json(payload)
            else:
                st.error(f"Error from backend API ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Connection failed: {e}")

# ------------------------------- BATCH --------------------------------
with tab2:
    st.subheader("Batch Predictions via CSV Upload")
    st.caption("CSV must contain these columns: " + ", ".join(FEATURES))
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.dataframe(df.head())

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        elif st.button("Run Batch Inference", type="primary"):
            data_payload = df[FEATURES].to_dict(orient="records")
            try:
                res = requests.post(
                    f"{BACKEND_URL}/predict_batch", json={"data": data_payload}, timeout=60
                )
                if res.status_code == 200:
                    df["Predicted_Sales"] = res.json().get("predictions", [])
                    st.success(f"Batch prediction complete: {len(df)} rows scored.")
                    st.dataframe(df)
                    st.download_button(
                        "Download predictions (CSV)",
                        df.to_csv(index=False).encode("utf-8"),
                        file_name="superkart_batch_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error(f"Batch prediction error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")
