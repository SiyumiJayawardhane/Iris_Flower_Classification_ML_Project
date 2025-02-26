import streamlit as st
import joblib
import numpy as np

# Load trained models
log_reg = joblib.load("logistic_regression_model.pkl")
knn = joblib.load("knn_model.pkl")

# Mapping numerical labels to species names
species_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

st.title("Iris Flower Species Prediction 🌺")

# User Inputs with automatic decimal conversion
sepal_length = st.number_input("Sepal Length (cm)", value=None, step=0.01, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", value=None, step=0.01, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", value=None, step=0.01, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", value=None, step=0.01, format="%.2f")

if st.button("Predict"):
    if None in [sepal_length, sepal_width, petal_length, petal_width]:
        st.error("Please enter valid numbers for all fields.")
    else:
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predictions
        log_reg_prediction = log_reg.predict(user_input)[0]
        knn_prediction = knn.predict(user_input)[0]

        log_reg_species = species_mapping[log_reg_prediction]
        knn_species = species_mapping[knn_prediction]

        st.write(f"**Logistic Regression Prediction:** {log_reg_species}")
        st.write(f"**KNN Prediction:** {knn_species}")

        if log_reg_species == knn_species:
            st.success(f"**Final Prediction: {log_reg_species} ✅**")
        else:
            st.warning("Models Disagree! Consider reviewing predictions.")
