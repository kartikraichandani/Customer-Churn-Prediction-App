# Customer-Churn-Prediction-App
This repository contains a Streamlit web application that predicts customer churn using a trained Artificial Neural Network (ANN) model. The model is trained on a dataset of customer information, including demographics, financial details, and service usage.  The app allows users to input customer data and receive a churn probability prediction.

## Repository Structure
customer-churn-prediction/
├── app.py           # Streamlit web application
├── model.h5         # Trained ANN model
├── label_encoder_gender.pkl # Saved LabelEncoder
├── onehot_encoder_geo.pkl  # Saved OneHotEncoder
├── scalar.pkl       # Saved StandardScaler
├── Churn_Modelling.csv # Dataset used for training
└── README.md        # This file

## Requirements

Includes a `requirements.txt` file in the directory containing the following:

streamlit
tensorflow
pandas
scikit-learn
numpy
pickle

## Model Training

The `Churn_Modelling.csv` dataset is used to train the ANN model. The training process involves:

1.  Data preprocessing: Handling categorical features (Gender, Geography) using Label Encoding and One-Hot Encoding, scaling numerical features using StandardScaler.
2.  Model building: Creating a sequential ANN model with Dense layers and ReLU activation functions.
3.  Model compilation: Using the Adam optimizer and binary cross-entropy loss function.
4.  Model training: Training the model on the training data and validating it on the test data.
5.  Saving the trained model and preprocessing objects (scalers, encoders).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
