from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from src.config import * 
    
def xgboost_model(X_train, y_train, X_test):
    """
    Ini untuk build model XGBoost
    """
    xgb_model = XGBRegressor(
        # parameters
    )

    # Train XGBoost pakai MultiOutputRegressor
    xgb = MultiOutputRegressor(xgb_model)
    xgb.fit(X_train, y_train)                       # training
    y_pred = xgb.predict(X_test)                    # inference
    
    return xgb, y_pred                              # balikin model & prediksi