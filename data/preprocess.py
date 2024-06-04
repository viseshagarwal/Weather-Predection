import pandas as pd


def preprocess_data(data):
    all_days_data = []
    for daily_data in data:
        forecast = daily_data.get("forecast", {}).get("forecastday", [])
        if forecast:
            day_data = forecast[0].get("day", {})
            all_days_data.append(
                {
                    "temp_c": day_data.get("avgtemp_c", None),
                    "humidity": day_data.get("avghumidity", None),
                }
            )

    df = pd.DataFrame(all_days_data)
    df.fillna(method="ffill", inplace=True)
    return df
