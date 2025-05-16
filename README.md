# SimpleTradingBotPython

A simple neural network built with PyTorch to generate **Buy/Sell** signals for stock trading using historical price data. This project demonstrates how to apply machine learning to financial time-series data with real-world data from Yahoo Finance.

## Project Overview

This bot uses a feedforward neural network to learn patterns from:
- Daily **closing prices**
- **50-day** and **200-day** moving averages
- Daily **percentage price changes**

It then predicts whether the next day's closing price will increase (Buy) or decrease (Sell).

## Features

- Fetches real-time stock data using `yfinance`
- Performs rolling average and price change feature engineering
- Scales features using `StandardScaler`
- Implements a PyTorch-based neural network classifier
- Outputs Buy/Sell decisions for each day in the dataset

## Model Architecture

- **Input layer**: 4 features (Close, 50 MA, 200 MA, Price Change)
- **Hidden layers**: 64 → 32 units with ReLU activation
- **Output layer**: 2 classes (Buy/Sell)
