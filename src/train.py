"""
TRAINING
===============
Melatih beberapa model regresi PER-NODE pada data peramalan gas yang telah diproses.
Setiap node mendapatkan 4 model terpisah (LR, RFR, XGBoost, SVR).
Total model: 4 algoritma x 6 node = 24 model.

Semua metrik, parameter, dan model di-log ke MLflow.
"""

import mlflow
import mlflow.sklearn
import mlflow.data.pandas_dataset
import pandas as pd
from mlflow.models import infer_signature
from mlflow import MlflowClient
from src.config import (
    RANDOM_STATE, TEST_SIZE, FEATURES, TARGET_COL, NUM_NODES,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
)
from sklearn.model_selection import train_test_split
from src.models.lr import linear_regression_model
from src.models.rfr import random_forest_regressor_model
from src.models.xgboost import xgboost_model
from src.models.svr import support_vector_regression_model
from src.evaluation.metrics import *
from src.config import *


# Mapping nama internal -> nama tampilan
MODEL_DISPLAY = {
    "lr": "Linear Regression",
    "rfr": "Random Forest Regressor",
    "xgbr": "XGBoost",
    "svr": "SVR",
}


def _train_single_node(X_train, y_train, X_val, y_val, X_test, y_test, node_id):
    """
    Melatih 4 model pada data dari satu node, log ke MLflow,
    dan mengembalikan model beserta metriknya.

    Setiap model dicatat sebagai satu MLflow run dengan:
      - Parameters: model_type, node_id, test_size, random_state, n_features
      - Metrics: train_mae, train_rmse, train_r2, val_mae, ..., test_mae, test_rmse, test_r2
      - Artifact: model disimpan via mlflow.sklearn.log_model()
    """
    results = {"models": {}, "metrics": {}}

    # Fungsi-fungsi pembangun model
    model_builders = {
        "lr": linear_regression_model,
        "rfr": random_forest_regressor_model,
        "xgbr": xgboost_model,
        "svr": support_vector_regression_model,
    }

    for model_key, builder_fn in model_builders.items():
        display_name = MODEL_DISPLAY[model_key]
        run_name = f"node{node_id}_{model_key}"

        with mlflow.start_run(run_name=run_name):
            # -- Log parameters --
            mlflow.log_param("model_type", model_key)
            mlflow.log_param("model_name", display_name)
            mlflow.log_param("node_id", node_id)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_val_samples", len(X_val))
            mlflow.log_param("n_test_samples", len(X_test))

            # -- Log datasets --
            train_df = pd.concat([X_train, y_train], axis=1)
            val_df = pd.concat([X_val, y_val], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            train_dataset = mlflow.data.from_pandas(
                train_df, name=f"node{node_id}_train"
            )
            val_dataset = mlflow.data.from_pandas(
                val_df, name=f"node{node_id}_val"
            )
            test_dataset = mlflow.data.from_pandas(
                test_df, name=f"node{node_id}_test"
            )

            mlflow.log_input(train_dataset, context="training")
            mlflow.log_input(val_dataset, context="validation")
            mlflow.log_input(test_dataset, context="testing")
            mlflow.set_tag("target_columns", "h2s,so2")

            # -- Train model --
            model = builder_fn(X_train, y_train)

            # -- Log hyperparameters model --
            model_params = model.get_params()
            for param_name, param_value in model_params.items():
                # Skip parameter yang berupa objek estimator (tidak bisa di-log)
                if hasattr(param_value, "fit"):
                    continue
                mlflow.log_param(f"hp_{param_name}", param_value)

            # -- Evaluasi --
            train_met = metrics(y_train, model.predict(X_train))
            val_met = val_metrics(y_val, model.predict(X_val))
            test_met = metrics(y_test, model.predict(X_test))

            # -- Log metrics (train) --
            mlflow.log_metric("train_mae", train_met["mae"])
            mlflow.log_metric("train_mse", train_met["mse"])
            mlflow.log_metric("train_rmse", train_met["rmse"])
            mlflow.log_metric("train_r2", train_met["r2"])

            # -- Log metrics (val) --
            mlflow.log_metric("val_mae", val_met["val_mae"])
            mlflow.log_metric("val_mse", val_met["val_mse"])
            mlflow.log_metric("val_rmse", val_met["val_rmse"])
            mlflow.log_metric("val_r2", val_met["val_r2"])

            # -- Log metrics (test) --
            mlflow.log_metric("test_mae", test_met["mae"])
            mlflow.log_metric("test_mse", test_met["mse"])
            mlflow.log_metric("test_rmse", test_met["rmse"])
            mlflow.log_metric("test_r2", test_met["r2"])

            # -- Log model artifact (dengan signature/schema) --
            y_pred_train = model.predict(X_train)
            signature = infer_signature(X_train, y_pred_train)

            model_info = mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"{model_key}_node{node_id}"
            )

            # -- Run Tags --
            mlflow.set_tag("node", str(node_id))
            mlflow.set_tag("algorithm", model_key)

            # -- Registered Model Tags --
            client = MlflowClient()
            reg_model_name = f"{model_key}_node{node_id}"

            client.set_registered_model_tag(
                reg_model_name, "node", str(node_id)
            )
            client.set_registered_model_tag(
                reg_model_name, "algorithm", model_key
            )
            client.set_registered_model_tag(
                reg_model_name, "algorithm_name", display_name
            )
            client.set_registered_model_tag(
                reg_model_name, "target_columns", "h2s,so2"
            )
            client.set_registered_model_tag(
                reg_model_name, "task", "multi-output-regression"
            )

            # -- Model Version Tags --
            latest_version = client.get_latest_versions(
                reg_model_name
            )[0].version
            client.set_model_version_tag(
                reg_model_name, latest_version, "test_r2",
                f"{test_met['r2']:.4f}"
            )
            client.set_model_version_tag(
                reg_model_name, latest_version, "test_rmse",
                f"{test_met['rmse']:.4f}"
            )

        # Simpan ke results dict
        results["models"][model_key] = model
        results["metrics"][model_key] = {
            "train": train_met,
            "val": val_met,
            "test": test_met,
        }

    return results


def train_model(df):
    """
    Melatih model regresi PER-NODE pada DataFrame yang disediakan.

    Untuk setiap node (1-6), data node tersebut di-split menjadi:
      - 70% training
      - 15% validasi
      - 15% testing

    Kemudian 4 algoritma dilatih dan di-log ke MLflow.

    Args:
        df: DataFrame yang sudah diproses (berisi kolom 'node', FEATURES, TARGET_COL).

    Returns:
        dict: {node_id: {"models": {...}, "metrics": {...}}}
    """
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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

        # Latih 4 model + log ke MLflow
        all_results[node_id] = _train_single_node(
            X_train, y_train, X_val, y_val, X_test, y_test, node_id
        )

    return all_results