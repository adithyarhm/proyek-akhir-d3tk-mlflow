from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from src.config import * 
    
def xgboost_model(X_train, y_train):
    """
    Ini untuk build model XGBoost
    """
    xgb_model = XGBRegressor(
        # parameters
    )

    # Train XGBoost pakai MultiOutputRegressor
    xgb = MultiOutputRegressor(xgb_model)
    xgb.fit(X_train, y_train)                       # training
    
    return xgb                                      # balikin model