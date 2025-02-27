import streamlit as st
import joblib
import numpy as np
import base64

# Load trained models
log_reg = joblib.load("web/logistic_regression_model.pkl")
knn = joblib.load("web/knn_model.pkl")

# Mapping numerical labels to species names
species_mapping = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

# Function to encode image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

background_base64 = get_base64_image(r"web/bg.png")

# Custom CSS for styling
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{background_base64}");
            background-size: cover;
        }}
        .stTitle {{
            background: linear-gradient(to right, rgba(0, 0, 0, 0.2), rgba(45, 45, 45, 0.2)); /* Dark gradient */
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            margin-top: -20px;  /* Fixes extra space */
        }}
        .block-container {{
            padding-top: 0px !important; /* Removes unwanted space */
        }}
        .stTextInput, .stNumberInput, .stButton, .stMarkdown {{
            background-color: #B2B2B2; 
            padding: 10px;
            border-radius: 10px;
        }}
        .stAlert {{
            color: black !important;
            font-weight: bold;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="stTitle">Iris Flower Species Prediction 🌺</h1>', unsafe_allow_html=True)

# User Inputs
sepal_length = st.number_input("Sepal Length (cm)", value=None, step=0.01, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", value=None, step=0.01, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", value=None, step=0.01, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", value=None, step=0.01, format="%.2f")

# Prediction button
if st.button("Predict"):
    if None in [sepal_length, sepal_width, petal_length, petal_width]:
        st.markdown('<p class="stAlert" style="background-color: #FFCCCB;"> Please enter valid numbers for all fields.</p>', unsafe_allow_html=True)
    else:
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Predictions
        log_reg_prediction = log_reg.predict(user_input)[0]
        knn_prediction = knn.predict(user_input)[0]

        log_reg_species = species_mapping[log_reg_prediction]
        knn_species = species_mapping[knn_prediction]

        st.write(f"**Logistic Regression Prediction:** {log_reg_species}")
        st.write(f"**KNN Prediction:** {knn_species}")

        # Display results with better visibility
        if log_reg_species == knn_species:
            st.markdown(f'<p class="stAlert" style="background-color: #90EE90;">**Final Prediction: {log_reg_species}**</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="stAlert" style="background-color: #FFD700;"> Models Disagree! Consider reviewing predictions.</p>', unsafe_allow_html=True)
