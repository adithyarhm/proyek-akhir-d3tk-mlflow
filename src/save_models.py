import pickle
import os
from src.config import *

save_dir = MODEL_SAVE_DIR

def save_trained_models(lr, rfr, xgbr, svr):
    """
    Menyimpan model-model yang sudah dilatih ke dalam format .pkl menggunakan modul pickle.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(os.path.join(save_dir, "linear_regression.pkl"), "wb") as f:
        pickle.dump(lr, f)
        
    with open(os.path.join(save_dir, "random_forest_regressor.pkl"), "wb") as f:
        pickle.dump(rfr, f)
        
    with open(os.path.join(save_dir, "xgboost_regressor.pkl"), "wb") as f:
        pickle.dump(xgbr, f)
        
    with open(os.path.join(save_dir, "support_vector_regression.pkl"), "wb") as f:
        pickle.dump(svr, f)
