from src.config import RANDOM_STATE
from src.config import TEST_SIZE
from sklearn.model_selection import train_test_split
from src.models.lr import linear_regression_model
from src.models.rfr import random_forest_regressor_model
from src.models.xgboost import xgboost_model
from src.models.svr import support_vector_regression_model
from src.evaluation.metrics import *
from src.config import *

def train_model(df):
    X = df[FEATURES].drop(columns=TARGET_COL)
    y = df[TARGET_COL]

    # Split data: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
        )

    
    ## Linear Regression
    # 1. Train model hanya dengan training set
    lr = linear_regression_model(X_train, y_train)
    # 2. Evaluasi train
    lr_train_metrics = metrics(y_train, lr.predict(X_train))
    # 3. Evaluasi validation
    lr_val_metrics = val_metrics(y_val, lr.predict(X_val))
    # 4. Evaluasi final pada test set
    lr_final_metrics = metrics(y_test, lr.predict(X_test))

    ## Random Forest Regression
    # 1. Train model hanya dengan training set
    rfr = random_forest_regressor_model(X_train, y_train)
    # 2. Evaluasi train
    rfr_train_metrics = metrics(y_train, rfr.predict(X_train))
    # 3. Evaluasi validation
    rfr_val_metrics = val_metrics(y_val, rfr.predict(X_val))
    # 4. Evaluasi final pada test set
    rfr_final_metrics = metrics(y_test, rfr.predict(X_test))

    ## XGBoost Regressor
    # 1. Train model hanya dengan training set
    xgbr = xgboost_model(X_train, y_train)
    # 2. Prediksi untuk masing-masing split
    # 3. Evaluasi train
    xgbr_train_metrics = metrics(y_train, xgbr.predict(X_train))
    # 4. Evaluasi validation
    xgbr_val_metrics = val_metrics(y_val, xgbr.predict(X_val))
    # 5. Evaluasi final pada test set
    xgbr_final_metrics = metrics(y_test, xgbr.predict(X_test))

    ## Support Vector Regression
    # 1. Train model hanya dengan training set
    svr = support_vector_regression_model(X_train, y_train)
    # 2. Evaluasi train
    svr_train_metrics = metrics(y_train, svr.predict(X_train))
    # 3. Evaluasi validation
    svr_val_metrics = val_metrics(y_val, svr.predict(X_val))
    # 4. Evaluasi final pada test set
    svr_final_metrics = metrics(y_test, svr.predict(X_test))

    models = (lr, rfr, xgbr, svr)
    final_metrics = (lr_final_metrics, rfr_final_metrics, xgbr_final_metrics, svr_final_metrics)
    
    return models, final_metrics