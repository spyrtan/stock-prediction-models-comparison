# Stock Prediction Models Comparison
Comparative analysis of ARIMA, LSTM, XGBoost, CNN, and Transformer models for stock price prediction.
![image](https://github.com/user-attachments/assets/48aa1c22-bfbc-4dc5-90f7-2e0fb0a45844)
# Project Directory Structure

The structure of this project is organized in a way that allows efficient storage, processing, and evaluation of stock prediction models. Here’s a breakdown of each folder:

## `data/`
This folder contains the raw and processed data used in model training.

- **`raw/`** – Contains data directly obtained from external sources (e.g., Yahoo Finance API, web scraping, etc.). You can find files in formats like `.csv`, `.json`, or `.pickle` here.
  
- **`processed/`** – Contains the data that has been cleaned and transformed to be ready for training models. This includes normalized data, feature engineering, and time-series processing.

## `models/`
In this folder, we save the trained models to avoid retraining them every time. This makes it easier to use a pre-trained model without repeating the training process.

Examples of files that might go here:

- `lstm_model.h5` (Keras model for LSTM)
- `xgboost_model.json` (XGBoost model in JSON format)
- `arima_model.pkl` (Pickle file for ARIMA model)

## `notebooks/`
This directory is for experimental work in Jupyter notebooks.

A typical file in this folder could be:

- **`EDA.ipynb`** – For Exploratory Data Analysis

## `src/`
This folder contains the main logic of the project, with Python scripts for preprocessing, training, evaluating, and saving models.

- **`preprocess.py`** – Contains functions for:
  - Fetching data (e.g., from APIs)
  - Cleaning data (e.g., removing NaN values)
  - Creating features (e.g., lag features, rolling means)
  - Normalizing data
  
  Key function:
  - `prepare_data()` → Returns training and test sets: `X_train`, `y_train`, `X_test`, `y_test`.

- **`train.py`** – This script includes the abstract function `train_model(model, X, y)` to train models. 
  - Alternatively, there can be a separate function for each model (e.g., `train_lstm_model()`).

- **`evaluate.py`** – Evaluates models and computes various metrics like:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - R² (R-squared)

## `results/`
This folder holds the outputs and results of the project, including:
- **Tables** comparing model performances (e.g., `.csv`).
- **Charts** and **plots** (e.g., `.png` files).
- **Metric reports** and analyses.

## `main.py`
The entry point of the project:
- Loading and preparing data
- Training the models
- Evaluating results
- Saving the outputs

## `README.md`
This file explains the project in detail:
- **Project description** – A brief explanation of the purpose and scope of the project.
- **How to run** – Instructions on setting up and running the project.
- **Requirements** – List of dependencies (e.g., `pip install -r requirements.txt`).
- **Models used** – An overview of the models included in the project and their intended purpose.
