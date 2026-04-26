from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from src.config import * 
    
def support_vector_regression_model(X_train, y_train):
    """
    Ini untuk build model Support Vector Regression
    """
    svr_model = SVR(
        # parameters
    )

    # Train SVR pakai MultiOutputRegressor
    svr = MultiOutputRegressor(svr_model)
    svr.fit(X_train, y_train)                       # training
    
    return svr                                      # balikin model