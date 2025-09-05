import streamlit as st
import numpy as np
import pickle
import base64
from tensorflow.keras.models import load_model

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

# ---------------------------
# BACKGROUND IMAGE FUNCTION
# ---------------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .block-container {{
        background: rgba(255,255,255,0.8);
        padding: 2rem;
        border-radius: 15px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call background
add_bg_from_local("churnpred.png")

# ---------------------------
# LOAD MODEL & PREPROCESSORS
# ---------------------------
model = load_model("model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder_gender.pkl", "rb") as f:
    le_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    ohe_geo = pickle.load(f)

# ---------------------------
# APP UI
# ---------------------------
st.title("📊 Customer Churn Prediction App")
st.markdown("Predict whether a customer will leave the bank or not.")

# Get possible geographies from encoder
geo_options = ohe_geo.categories_[0].tolist()

# User inputs
geography = st.selectbox("🌍 Select Geography", geo_options)
gender = st.selectbox("👤 Select Gender", le_gender.classes_.tolist())
age = st.slider("🎂 Age", 18, 92, 30)
balance = st.number_input("💰 Balance", min_value=0.0, value=50000.0, step=1000.0)
credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input("💵 Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)
tenure = st.slider("📅 Tenure (Years with Bank)", 0, 10, 5)
num_products = st.selectbox("📦 Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("💳 Has Credit Card?", [0, 1])
is_active_member = st.selectbox("✅ Is Active Member?", [0, 1])

# ---------------------------
# DATA PROCESSING
# ---------------------------
# Encode inputs same as training
geo_encoded = ohe_geo.transform([[geography]]).toarray()
gender_encoded = le_gender.transform([gender])[0]

features = np.concatenate((geo_encoded,
                           np.array([[credit_score, gender_encoded, age, tenure, balance,
                                      num_products, has_cr_card, is_active_member,
                                      estimated_salary]])), axis=1)

# Scale
features_scaled = scaler.transform(features)

# ---------------------------
# PREDICTION
# ---------------------------
if st.button("🔮 Predict"):
    prob = model.predict(features_scaled)[0][0]
    result = "❌ Customer is likely to churn!" if prob > 0.5 else "✅ Customer will stay."
    st.subheader(result)
    st.write(f"**Churn Probability:** {prob:.2f}")
