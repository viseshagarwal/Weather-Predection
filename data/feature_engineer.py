def feature_engineering(df):
    df["temp_diff"] = df["temp_c"].diff()
    df["humidity_diff"] = df["humidity"].diff()
    df.dropna(inplace=True)
    return df
