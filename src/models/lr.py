from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from src.config import * 
    
def linear_regression_model(X_train, y_train, X_test):
    """
    Ini untuk build model Linear Regression
    """
    lr_model = LinearRegression(
        # parameters
    )

    # Train Linear Regression pakai MultiOutputRegressor
    lr = MultiOutputRegressor(lr_model)
    lr.fit(X_train, y_train)                  # training
    y_pred = lr.predict(X_test)               # inference
    
    return lr, y_pred                         # balikin model & prediksi
    