"""
MAIN PROGRAM
=================
Seluruh pipeline ML (per-node) dengan MLflow tracking:
  1. Pengambilan Data   - memuat & menggabungkan CSV node mentah
  2. Pra-pemrosesan     - penanganan outlier + feature engineering
  3. Pelatihan          - melatih 4 model per node + log ke MLflow
  4. Evaluasi           - menampilkan metrik akhir pada data uji per-node
"""


from src.config import *
from src.data.data_loader import load_and_merge_raw_nodes, load_data
from src.preprocess import preprocess_pipeline
from src.train import train_model
import pandas as pd


MODEL_DISPLAY_NAMES = {
    "lr": "Linear Regression",
    "rfr": "Random Forest Regressor",
    "xgbr": "XGBoost",
    "svr": "SVR",
}


def main():
    # -- Step 1: Pengambilan Data
    print("=" * 60)
    print("STEP 1: Memuat dan menggabungkan data mentah dari node ...")
    print("=" * 60)
    df_raw = load_and_merge_raw_nodes()
    print(f"  -> Bentuk data gabungan: {df_raw.shape}")


    # -- Step 2: Preprocessing & Feature Engineering
    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing (penanganan outlier + Feature Engineering) ...")
    print("=" * 60)
    df_processed = preprocess_pipeline(df_raw)
    print(f"  -> Bentuk data setelah proses: {df_processed.shape}")

    # Simpan data hasil pemrosesan untuk inspeksi / caching
    df_processed.to_csv(NODE_COMBINED_FINAL, index=False)
    print(f"  -> Data hasil pemrosesan disimpan ke {NODE_COMBINED_FINAL}")


    # -- Step 3: Training Per-Node + MLflow Tracking
    print("\n" + "=" * 60)
    print("STEP 3: Training model per-node (+ MLflow tracking) ...")
    print("=" * 60)

    # Menghapus baris dengan NaN akibat fitur diff/ratio
    df_clean = df_processed.dropna(subset=FEATURES)
    print(f"  -> Bentuk data latih bersih: {df_clean.shape}")

    all_results = train_model(df_clean)


    # -- Step 4: Evaluasi Per-Node
    print("\n" + "=" * 60)
    print("STEP 4: Metrik Akhir pada Data Uji (Per-Node)")
    print("=" * 60)

    for node_id in sorted(all_results.keys()):
        node_data = all_results[node_id]
        print(f"\n  {'='*50}")
        print(f"  NODE {node_id}")
        print(f"  {'='*50}")

        for model_key, display_name in MODEL_DISPLAY_NAMES.items():
            if model_key in node_data["metrics"]:
                test_met = node_data["metrics"][model_key]["test"]
                print(f"    {display_name}:")
                print(f"      MAE={test_met['mae']:.4f}  "
                      f"RMSE={test_met['rmse']:.4f}  "
                      f"R2={test_met['r2']:.4f}")


    print("\n" + "=" * 60)
    print("Pipeline selesai [OK]")
    print("=" * 60)
    print("\n  Jalankan 'mlflow ui' untuk melihat dashboard MLflow.")
    print("  Buka http://localhost:5000 di browser.")


if __name__ == "__main__":
    main()