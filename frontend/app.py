import streamlit as st
import requests
import pandas as pd

# -------------------------------------------------------------------
# REPLACE THIS WITH YOUR LIVE PORT 7860 CODESPACES URL (NO TRAILING SLASH)
# -------------------------------------------------------------------
BACKEND_URL = "https://curly-enigma-jrg444vx74qc5x5x-7860.app.github.dev"

st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("🛒 SuperKart Sales Prediction Dashboard")

st.markdown("---")

tab1, tab2 = st.tabs(["Online Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Single Product Sales Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input("Product Weight", value=12.5)
        visibility = st.number_input("Product Visibility", value=0.016, format="%.3f")
        price = st.number_input("Product Price", value=141.6)
        year = st.number_input("Supermarket Opening Year", value=1999, step=1)
        
    with col2:
        size = st.selectbox("Supermarket Size", ["Small", "Medium", "High"])
        location = st.selectbox("Supermarket Location Type", ["Tier 1", "Tier 2", "Tier 3"])
        outlet_type = st.selectbox("Supermarket Type", [
            "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"
        ])
        
    if st.button("Predict Item Outlet Sales"):
        payload = {
            "Product_Weight": weight,
            "Product_Visibility": visibility,
            "Product_Price": price,
            "Supermarket_Opening_Year": year,
            "Supermarket_Size": size,
            "Supermarket_Location_Type": location,
            "Supermarket_Type": outlet_type
        }
        
        try:
            res = requests.post(f"{BACKEND_URL}/predict", json=payload)
            if res.status_code == 200:
                pred = res.json().get("prediction", 0)
                st.success(f"Predicted Sales Amount: **${pred:,.2f}**")
            else:
                st.error(f"Error from backend API ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")

with tab2:
    st.subheader("Batch Predictions via CSV Upload")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())
        
        if st.button("Run Batch Inference"):
            data_payload = df.to_dict(orient="records")
            try:
                res = requests.post(f"{BACKEND_URL}/predict_batch", json={"data": data_payload})
                if res.status_code == 200:
                    predictions = res.json().get("predictions", [])
                    df["Predicted_Sales"] = predictions
                    st.success("Batch Prediction Complete!")
                    st.dataframe(df)
                else:
                    st.error(f"Batch prediction error ({res.status_code}): {res.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")
