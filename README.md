# House-Prediction

## Project Overview
This project predicts house prices based on features such as area, bedrooms, bathrooms, location, and more. The best regression model (Linear, Ridge, or Lasso) is trained and saved, and a Streamlit web app is used for making real-time predictions.

---

## Dataset
- Source: Kaggle "Housing Prices" dataset  
- Number of rows: 545  
- Features:
  - price (target)
  - area, bedrooms, bathrooms, stories
  - mainroad, guestroom, basement, hotwaterheating, airconditioning
  - parking, prefarea, furnishingstatus

---

## Project Structure
House-Prediction/
│
├─ app.py # Streamlit app
├─ train.py # Script for training models
├─ house_price_model.joblib # Saved model
├─ Housing.csv # Original dataset
├─ heatmap.png # Correlation heatmap
└─ README.md # This file

## Requirements
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
Pillow
