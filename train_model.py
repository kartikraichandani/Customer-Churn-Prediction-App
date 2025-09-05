import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("Churn_Modelling.csv")

# Encode Gender
le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])

# Encode Geography
ohe_geo = OneHotEncoder(handle_unknown='ignore')
geo_encoded = ohe_geo.fit_transform(data[['Geography']]).toarray()

# Build feature matrix
X = np.concatenate((geo_encoded,
                    data[['CreditScore', 'Gender', 'Age', 'Tenure',
                          'Balance', 'NumOfProducts', 'HasCrCard',
                          'IsActiveMember', 'EstimatedSalary']].values), axis=1)

y = data['Exited'].values

print("✅ Training features shape:", X.shape)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build simple ANN
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model + preprocessors
model.save("model.h5")

with open("label_encoder_gender.pkl", "wb") as f:
    pickle.dump(le_gender, f)

with open("onehot_encoder_geo.pkl", "wb") as f:
    pickle.dump(ohe_geo, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("🎉 Model and preprocessors saved successfully!")
