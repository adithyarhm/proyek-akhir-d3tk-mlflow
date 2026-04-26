from sklearn.model_selection import train_test_split
from src.models.lr import linear_regression_model
from src.models.rfr import random_forest_regressor_model
from src.models.xgboost import xgboost_model
from src.evaluation.metrics import metrics
from src.config import *

def train_model(df):
    X = df[FEATURES].drop(columns=TARGET_COL)
        
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Train Linear Regression
    lr, lr_y_pred = linear_regression_model(X_train, y_train, X_test)
    lr_metrics = metrics(y_test, lr_y_pred)

    # Train Random Forest Regressor
    rfr, rfr_y_pred = random_forest_regressor_model(X_train, y_train, X_test)
    rfr_metrics = metrics(y_test, rfr_y_pred)

    # # Train XGBoost
    xgb, xgb_y_pred = xgboost_model(X_train, y_train, X_test)
    xgb_metrics = metrics(y_test, xgb_y_pred)
    
    return lr_metrics, rfr_metrics, xgb_metrics