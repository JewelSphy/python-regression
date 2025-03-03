import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np


Housing_data= pd.read_csv('Housing.csv')
print(Housing_data.head())
print(Housing_data.dtypes)
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in categorical_columns:
    Housing_data[col] = Housing_data[col].map({'yes': 1, 'no': 0})

# Convert furnishing status to numerical values to avoid errors 
Housing_data['furnishingstatus'] = Housing_data['furnishingstatus'].map({'furnished': 2, 'semi-furnished': 1, 'unfurnished': 0})

# Create a 'total_rooms' column
Housing_data['total_rooms'] = Housing_data['bedrooms'] + Housing_data['bathrooms'] + Housing_data['stories']

# Select features and target
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'total_rooms',
            'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
X = Housing_data[features]
y = Housing_data['price']

# Splitting the  data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test = y_scaler.transform(y_test.values.reshape(-1, 1))

# Building the model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Output layer
])


optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# this prevents overfitting
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# training the model
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, callbacks=[early_stopping], verbose=1)


loss, mae = model.evaluate(X_test, y_test)


y_pred = y_scaler.inverse_transform(model.predict(X_test))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {loss}")

y_test_original = y_scaler.inverse_transform(y_test.reshape(-1, 1))  
y_pred_original = y_pred.reshape(-1, 1) 
# Scatter plot
plt.figure(figsize=(8, 6))

# Scatter plot
plt.scatter(y_test_original, y_pred_original, color='blue', alpha=0.5, s=20, label='Predicted vs Actual')

# Prediction line
min_val = min(y_test_original.min(), y_pred_original.min())
max_val = max(y_test_original.max(), y_pred_original.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, linestyle='dashed', label='Perfect Prediction')

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Show plot
plt.show()

