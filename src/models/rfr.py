from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from src.config import * 
    
def random_forest_regressor_model(X_train, y_train, X_test):
    """
    Ini untuk build model Random Forest Regressor
    """
    rfr_model = RandomForestRegressor(
        # parameters
    )
    
    # Train Random Forest Regressor pakai MultiOutputRegressor
    rfr = MultiOutputRegressor(rfr_model)
    rfr.fit(X_train, y_train)                  # training
    y_pred = rfr.predict(X_test)               # inference

    return rfr, y_pred                         # balikin model & prediksi