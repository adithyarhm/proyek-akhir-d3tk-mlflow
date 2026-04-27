"""
SAVE MODELS
=================
Menyimpan model-model yang sudah dilatih per-node ke format .pkl.
Format nama file: nama_model_node{N}.pkl
"""

import pickle
import os
from src.config import MODEL_SAVE_DIR


# Mapping nama internal -> nama dasar file (tanpa node dan ekstensi)
MODEL_BASE_NAMES = {
    "lr": "linear_regression",
    "rfr": "random_forest_regressor",
    "xgbr": "xgboost_regressor",
    "svr": "support_vector_regression",
}


def save_trained_models(all_results):
    """
    Menyimpan semua model per-node ke dalam format .pkl.

    Format penamaan:
        saved-models/
            linear_regression_node1.pkl
            linear_regression_node2.pkl
            ...
            random_forest_regressor_node1.pkl
            ...
            xgboost_regressor_node6.pkl
            support_vector_regression_node6.pkl

    Args:
        all_results: dict dari train_model(), dengan struktur:
            {node_id: {"models": {"lr": model, ...}, "metrics": {...}}}
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    for node_id, node_data in sorted(all_results.items()):
        models = node_data["models"]
        for model_key, model in models.items():
            base_name = MODEL_BASE_NAMES.get(model_key, model_key)
            filename = f"{base_name}_node{node_id}.pkl"
            filepath = os.path.join(MODEL_SAVE_DIR, filename)
            with open(filepath, "wb") as f:
                pickle.dump(model, f)

        print(f"  -> Node {node_id}: {len(models)} model disimpan")

    print(f"  -> Semua model disimpan di {MODEL_SAVE_DIR}/")
