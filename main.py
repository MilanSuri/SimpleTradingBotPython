import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def fetch_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data['50_MA'] = data['Close'].rolling(window=50).mean()  # Averages the price over 50 days. Short-term trade indicator.
    data['200_MA'] = data['Close'].rolling(window=200).mean()  # Averages the price over 200 days. Long-term indicator.

    data['Price_Change'] = data['Close'].pct_change()  # Percentage change in closing price can represent volatility
    data.dropna(inplace=True)  # Removes null values created by rolling averages

    # Defines a target for when to buy or sell the stock: if the price is higher the next day, buy (1); if lower, sell (0).
    data['Target'] = (data['Price_Change'].shift(-1) > 0).astype(int)

    features = data[['Close', '50_MA', '200_MA', 'Price_Change']]  # Inputs for the neural network

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    target = data['Target']  # Buy/Sell labels
    return features_scaled, target

class TradingNN(nn.Module):
    def __init__(self, input_size):
        super(TradingNN, self).__init__()

        # Neural network structure - connected layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Sets output to two classes: 0 = Sell, 1 = Buy

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function for first layer
        x = torch.relu(self.fc2(x))  # Activation function for second layer
        x = self.fc3(x)  # Final output layer
        return x

def train_model(model, features, target, epochs=100):
    criterion = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer for weight updates

    # Convert numpy arrays to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    target_tensor = torch.tensor(target.values, dtype=torch.long)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()  # Reset gradients

        outputs = model(features_tensor)  # Forward pass
        loss = criterion(outputs, target_tensor)  # Compute loss

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def make_predictions(model, features):
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disables gradient computation to increase efficiency
        features_tensor = torch.tensor(features, dtype=torch.float32)  # Convert from numpy array to PyTorch tensor
        outputs = model(features_tensor)  # Pass data through the trained model
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score (Buy or Sell)
    return predicted  # Return the predicted labels (0 for Sell, 1 for Buy)

def run_trading_bot(symbol, start_date, end_date):
    # Fetch and preprocess the data
    data = fetch_data(symbol, start_date, end_date)
    features, target = preprocess_data(data)

    # Train the model
    input_size = features.shape[1]  # Number of features
    model = TradingNN(input_size)
    train_model(model, features, target, epochs=100)

    # Make predictions
    predictions = make_predictions(model, features)

    # Print buy or sell signals
    for i in range(len(predictions)):
        action = "Buy" if predictions[i] == 1 else "Sell"
        print(f"Day {i}: {action}")

run_trading_bot('TSLA', '2024-1-1', '2025-03-12')
