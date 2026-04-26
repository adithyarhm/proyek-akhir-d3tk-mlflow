from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from src.config import * 
    
def linear_regression_model(X_train, y_train):
    """
    Ini untuk Build dan train model Linear Regression dengan MultiOutputRegressor
    """
    lr_model = LinearRegression(
        # parameters
    )

    # Train Linear Regression pakai MultiOutputRegressor
    lr = MultiOutputRegressor(lr_model)
    lr.fit(X_train, y_train)                  # training
    
    return lr                                 # balikin model
    