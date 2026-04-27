"""
TRAINING
===============
Melatih beberapa model regresi PER-NODE pada data peramalan gas yang telah diproses.
Setiap node mendapatkan 4 model terpisah (LR, RFR, XGBoost, SVR).
Total model: 4 algoritma x 6 node = 24 model.
"""

from src.config import RANDOM_STATE, TEST_SIZE, FEATURES, TARGET_COL, NUM_NODES
from sklearn.model_selection import train_test_split
from src.models.lr import linear_regression_model
from src.models.rfr import random_forest_regressor_model
from src.models.xgboost import xgboost_model
from src.models.svr import support_vector_regression_model
from src.evaluation.metrics import *
from src.config import *


def _train_single_node(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Melatih 4 model pada data dari satu node dan mengembalikan model beserta metriknya.

    Returns:
        dict: {
            "models": {"lr": model, "rfr": model, "xgbr": model, "svr": model},
            "metrics": {
                "lr": {"train": {...}, "val": {...}, "test": {...}},
                "rfr": {...}, "xgbr": {...}, "svr": {...}
            }
        }
    """
    results = {"models": {}, "metrics": {}}

    # Linear Regression
    lr = linear_regression_model(X_train, y_train)
    results["models"]["lr"] = lr
    results["metrics"]["lr"] = {
        "train": metrics(y_train, lr.predict(X_train)),
        "val": val_metrics(y_val, lr.predict(X_val)),
        "test": metrics(y_test, lr.predict(X_test)),
    }

    # Random Forest Regressor
    rfr = random_forest_regressor_model(X_train, y_train)
    results["models"]["rfr"] = rfr
    results["metrics"]["rfr"] = {
        "train": metrics(y_train, rfr.predict(X_train)),
        "val": val_metrics(y_val, rfr.predict(X_val)),
        "test": metrics(y_test, rfr.predict(X_test)),
    }

    # XGBoost Regressor
    xgbr = xgboost_model(X_train, y_train)
    results["models"]["xgbr"] = xgbr
    results["metrics"]["xgbr"] = {
        "train": metrics(y_train, xgbr.predict(X_train)),
        "val": val_metrics(y_val, xgbr.predict(X_val)),
        "test": metrics(y_test, xgbr.predict(X_test)),
    }

    # Support Vector Regression
    svr = support_vector_regression_model(X_train, y_train)
    results["models"]["svr"] = svr
    results["metrics"]["svr"] = {
        "train": metrics(y_train, svr.predict(X_train)),
        "val": val_metrics(y_val, svr.predict(X_val)),
        "test": metrics(y_test, svr.predict(X_test)),
    }

    return results


def train_model(df):
    """
    Melatih model regresi PER-NODE pada DataFrame yang disediakan.

    Untuk setiap node (1-6), data node tersebut di-split menjadi:
      - 70% training
      - 15% validasi
      - 15% testing

    Kemudian 4 algoritma dilatih pada split tersebut.

    Args:
        df: DataFrame yang sudah diproses (berisi kolom 'node', FEATURES, TARGET_COL).

    Returns:
        dict: {
            node_id: {
                "models": {"lr": model, "rfr": model, "xgbr": model, "svr": model},
                "metrics": {
                    "lr": {"train": {...}, "val": {...}, "test": {...}},
                    "rfr": {...}, "xgbr": {...}, "svr": {...}
                }
            }
        }
    """
    feature_cols = [c for c in FEATURES if c not in TARGET_COL]
    all_results = {}

    for node_id in range(1, NUM_NODES + 1):
        print(f"\n  -- Training Node {node_id} --")

        # Filter data untuk node ini
        df_node = df[df["node"] == node_id].copy()

        if len(df_node) == 0:
            print(f"     [SKIP] Tidak ada data untuk Node {node_id}")
            continue

        X = df_node[feature_cols]
        y = df_node[TARGET_COL]

        # Split: 70% train, 15% val, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
        )

        print(f"     Data: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # Latih 4 model untuk node ini
        all_results[node_id] = _train_single_node(
            X_train, y_train, X_val, y_val, X_test, y_test
        )

    return all_results