from sklearn.metrics import mean_squared_error
from models.utils import create_dataset
import numpy as np


def evaluate_lstm_model(model, scaler, time_step, data):
    scaled_data = scaler.transform(data)
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    predictions = model.predict(X)

    # Inverse transform only the temperature column
    predictions = scaler.inverse_transform(
        np.concatenate(
            [predictions, np.zeros((predictions.shape[0], data.shape[1] - 1))], axis=1
        )
    )[:, 0]

    mse = mean_squared_error(y, predictions)
    return mse, predictions


def evaluate_rf_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions
