from sklearn.model_selection import train_test_split
import pandas as pd
from models.lstm_model import train_lstm_model
from models.random_forest_model import optimize_random_forest


def train_models(features):
    X = features[["temp_c", "humidity"]]
    y = features["temp_diff"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lstm_model, scaler, time_step = train_lstm_model(features.values)
    best_rf_model = optimize_random_forest(X_train, y_train)

    return lstm_model, scaler, time_step, best_rf_model, X_test, y_test
