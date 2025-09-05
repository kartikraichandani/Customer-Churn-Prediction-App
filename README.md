# Customer Churn Prediction App 📉🤖

This repository hosts a **Streamlit web application** that predicts **customer churn** using a pre-trained **Artificial Neural Network (ANN)** model. The model leverages customer data, including **demographics, financial info, and service usage**, to estimate the likelihood of churn. Users can simply input customer details and get a **churn probability prediction** instantly! ⚡

---

## Repository Structure 📂

customer-churn-prediction/
├── app.py                  # Streamlit web app
├── model.h5                # Trained ANN model
├── label_encoder_gender.pkl # Saved LabelEncoder
├── onehot_encoder_geo.pkl  # Saved OneHotEncoder
├── scalar.pkl              # Saved StandardScaler
├── Churn_Modelling.csv     # Dataset used for training
└── README.md               # This file

---

## Requirements 🛠️

Install the dependencies listed in **requirements.txt**:

- `streamlit`
- `tensorflow`
- `pandas`
- `scikit-learn`
- `numpy`
- `pickle`

---

## Model Training 🏋️‍♂️

The **Churn_Modelling.csv** dataset is used to train the ANN model. The process includes:

1. **Data Preprocessing** 🧹  
   - Handling categorical features (**Gender**, **Geography**) with **Label Encoding** and **One-Hot Encoding**.  
   - Scaling numerical features using **StandardScaler**.  

2. **Model Building** 🏗️  
   - Creating a **Sequential ANN** with **Dense layers** and **ReLU activations**.  

3. **Model Compilation** ⚙️  
   - Using **Adam optimizer** and **binary cross-entropy** loss.  

4. **Model Training** 📊  
   - Training on the dataset and validating on test data.  

5. **Saving Artifacts** 💾  
   - Saving the trained **model**, **scalers**, and **encoders** for later use.  

---

## Contributing 🤝

Contributions are always welcome! Feel free to **open an issue** or **submit a pull request**.
