from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def optimize_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
    }

    model = RandomForestRegressor()
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
