import streamlit as st
import joblib
import pandas as pd
import os
from PIL import Image

# -----------------------------
# LOAD MODEL SAFELY
# -----------------------------
model_path = os.path.join(os.path.dirname(__file__), "house_price_model.joblib")

if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please run train.py first!")
    st.stop()

model = joblib.load(model_path)

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("🏡 House Price Prediction App")
st.write("Enter the house features to predict the price.")

# Input fields matching the dataset columns
area = st.number_input("Area (sq ft)", min_value=100, max_value=10000, value=1000)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Stories", min_value=1, max_value=5, value=1)
mainroad = st.selectbox("Main Road", ["yes", "no"])
guestroom = st.selectbox("Guest Room", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwater = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
parking = st.number_input("Parking", min_value=0, max_value=5, value=1)
furnishing = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])
prefarea = st.selectbox("Preferred Area", ["yes", "no"])  # <-- added

# Arrange inputs into a DataFrame in the same column order as training
input_data = pd.DataFrame({
    "area": [area],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "stories": [stories],
    "mainroad": [mainroad],
    "guestroom": [guestroom],
    "basement": [basement],
    "hotwaterheating": [hotwater],
    "airconditioning": [airconditioning],
    "parking": [parking],
    "furnishingstatus": [furnishing],
    "prefarea": [prefarea]  # matches training column
})

# -----------------------------
# MAKE PREDICTION
# -----------------------------
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"Estimated House Price: ₦{prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.subheader("Correlation Heatmap")
heatmap_image = Image.open("correlation_heatmap.png")
st.image(heatmap_image, caption="Correlation Heatmap", use_column_width=True)

st.markdown("""
**Correlation Heatmap Interpretation**

The correlation heatmap shows how numerical variables in the housing dataset relate to each other and to the target variable, price.  

The strongest predictors of house price are **area (0.54)**, **bathrooms (0.52)**, and **stories (0.42)**, indicating that larger homes with more facilities generally have higher prices.  

**Bedrooms (0.37)** and **parking spaces (0.38)** also show moderate positive relationships with price.
""")