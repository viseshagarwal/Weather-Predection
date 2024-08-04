import streamlit as st
import pandas as pd
from data.fetch_data import fetch_weather_data
from data.preprocess import preprocess_data
from data.feature_engineer import feature_engineering
from models.train_models import train_models
from models.evaluate_models import evaluate_lstm_model, evaluate_rf_model
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()

st.title("Weather Prediction App")

api_key = os.getenv("WEATHER_API_KEY")

location = st.text_input("Enter location")

if st.button("Fetch Weather Data"):
    with st.spinner("Fetching and processing data..."):
        try:
            data = fetch_weather_data(api_key, location)
            # st.write(data)
            preprocessed_data = preprocess_data(data)
            # st.write(preprocessed_data)
            features = feature_engineering(preprocessed_data)
            # st.write(features)
            if len(features) < 20:
                st.error(
                    "Not enough data points to train the model. Please fetch more data."
                )
            else:
                lstm_model, scaler, time_step, best_rf_model, X_test, y_test = (
                    train_models(features)
                )

                lstm_mse, lstm_predictions = evaluate_lstm_model(
                    lstm_model, scaler, time_step, features.values
                )
                st.write(f"LSTM Model Mean Squared Error: {lstm_mse}")

                rf_mse, rf_predictions = evaluate_rf_model(
                    best_rf_model, X_test, y_test
                )
                st.write(f"Random Forest Model Mean Squared Error: {rf_mse}")

                predictions_df = pd.DataFrame(
                    {
                        "Actual": y_test,
                        "LSTM Predictions": lstm_predictions[: len(y_test)],
                        "RF Predictions": rf_predictions,
                    }
                )

                # Plot the actual vs. predicted values for both models
                st.subheader("Actual vs. Predicted Values")
                fig, ax = plt.subplots()
                sns.lineplot(data=predictions_df, ax=ax)
                plt.xlabel("Sample Index")
                plt.ylabel("Temperature (C)")
                plt.title("Actual vs. Predicted Values")
                st.pyplot(fig)

                # Plot the distribution of the errors
                st.subheader("Error Distribution")
                lstm_errors = (
                    predictions_df["Actual"] - predictions_df["LSTM Predictions"]
                )
                rf_errors = predictions_df["Actual"] - predictions_df["RF Predictions"]
                fig, ax = plt.subplots()
                sns.histplot(
                    lstm_errors, kde=True, ax=ax, label="LSTM Errors", color="blue"
                )
                sns.histplot(rf_errors, kde=True, ax=ax, label="RF Errors", color="red")
                plt.xlabel("Error (C)")
                plt.title("Error Distribution")
                plt.legend()
                st.pyplot(fig)

                # Show actual vs. predicted scatter plot for better error visualization
                st.subheader("Actual vs. Predicted Scatter Plot")
                fig, ax = plt.subplots()
                sns.scatterplot(
                    x=predictions_df["Actual"],
                    y=predictions_df["LSTM Predictions"],
                    ax=ax,
                    label="LSTM Predictions",
                    color="blue",
                )
                sns.scatterplot(
                    x=predictions_df["Actual"],
                    y=predictions_df["RF Predictions"],
                    ax=ax,
                    label="RF Predictions",
                    color="red",
                )
                plt.xlabel("Actual Temperature (C)")
                plt.ylabel("Predicted Temperature (C)")
                plt.title("Actual vs. Predicted Scatter Plot")
                plt.legend()
                st.pyplot(fig)

                # Display descriptive statistics
                st.subheader("Descriptive Statistics of Errors")
                st.write("LSTM Errors:")
                st.write(lstm_errors.describe())
                st.write("RF Errors:")
                st.write(rf_errors.describe())

                # Original Data Line Chart with more features
                st.subheader("Original Data Line Chart")
                fig, ax = plt.subplots()
                features.plot(ax=ax)
                plt.xlabel("Sample Index")
                plt.ylabel("Values")
                plt.title("Original Data Features")
                st.pyplot(fig)

                # Display the latest weather predictions
                st.subheader("Latest Weather Predictions")
                st.write(f"LSTM Model Prediction: {lstm_predictions[-1]:.2f}°C")
                st.write(f"Random Forest Model Prediction: {rf_predictions[-1]:.2f}°C")

                # Add predictions to the original data line chart
                st.subheader("Original Data and Predictions")
                fig, ax = plt.subplots()
                ax.plot(features.index, features["temp_c"], label="Actual Temperature")
                ax.plot(
                    features.index[-len(lstm_predictions) :],
                    lstm_predictions,
                    label="LSTM Predictions",
                    linestyle="--",
                )
                ax.plot(
                    features.index[-len(rf_predictions) :],
                    rf_predictions,
                    label="RF Predictions",
                    linestyle="--",
                )
                plt.xlabel("Sample Index")
                plt.ylabel("Temperature (C)")
                plt.title("Original Data and Predictions")
                plt.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing data: {e}")
