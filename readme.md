# Weather Prediction Application

This Weather Prediction Application fetches weather data for a specified location using an API, preprocesses and engineers features from the data, trains two machine learning models (LSTM and Random Forest), and visualizes the results, including predictions and errors.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features

- Fetches weather data from an external API.
- Preprocesses and engineers features from the fetched data.
- Trains LSTM and Random Forest models to predict weather conditions.
- Evaluates model performance using Mean Squared Error.
- Visualizes actual vs. predicted values, error distributions, and descriptive statistics.
- Provides interactive visualizations using Streamlit.

## Requirements

- Python 3.7 or higher
- Streamlit
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow (for LSTM model)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/weather-prediction-app.git
   cd weather-prediction-app
   ```
2. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```
3. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. Obtain an API key from the weather data provider (e.g., WeatherAPI).

2. Run the application:
   ```sh
   streamlit run app.py
   ```
3. Enter the API key and location to fetch weather data.

4. Click the "Fetch Weather Data" button to fetch and process the data.

5. View the predictions and visualizations.

## Project Structure

The project structure is as follows:

```
weather-prediction-app/
├── data/
│   ├── fetch_data.py
│   ├── preprocess.py
│   └── feature_engineer.py
├── models/
│   ├── train_models.py
│   └── evaluate_models.py
├── app.py
├── requirements.txt
└── README.md
```

- `data/fetch_data.py`: Contains the function to fetch weather data from the API.
- `data/preprocess.py`: Contains the function to preprocess the fetched data.
- `data/feature_engineer.py`: Contains the function to perform feature engineering.
- `models/train_models.py`: Contains functions to train the LSTM and Random Forest models.
- `models/evaluate_models.py`: Contains functions to evaluate the trained models.
- `app.py`: Main application script that ties everything together and runs the Streamlit app.
- `requirements.txt`: Lists the required Python packages.
- `README.md`: This file.
