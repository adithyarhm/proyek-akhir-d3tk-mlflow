"""
PREPROCESSING
====================
Handles proses capping outlier berbasis IQR per node dan feature engineering
(fitur waktu, lag/diff, rasio gas).
"""

import pandas as pd
import numpy as np
from src.config import NUMERIC_COLS, NUM_NODES


# Outlier Handling – Proses capping outlier berbasis IQR per node
def _cap_outliers_iqr(series: pd.Series, factor: float = 1.5) -> pd.Series:
    """
    Melakukan capping outlier pada sebuah Series menggunakan metode IQR.
    Nilai di bawah Q1 - factor*IQR diubah menjadi nilai batas bawah tersebut.
    Nilai di atas Q3 + factor*IQR diubah menjadi nilai batas atas tersebut.
    
    Args:
        series: pandas Series berisi nilai numerik.
        factor: Faktor pengali IQR (default 1.5).
    
    Returns:
        pd.Series: Series dengan nilai outlier telah dicapping.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower=lower, upper=upper)


def handle_outliers(df: pd.DataFrame,
                    numeric_cols: list[str] | None = None,
                    num_nodes: int | None = None) -> pd.DataFrame:
    """
    Menerapkan strategi capping outlier berbasis IQR pada setiap kolom numerik per node.
    Mengikuti alur yang digunakan pada notebook pra-pemrosesan (03_preprocessing).
    
    Data akan dipisah berdasarkan node, outlier di-capping per node per kolom,
    lalu seluruh hasilnya digabungkan kembali menjadi satu DataFrame.
    
    Args:
        df: DataFrame gabungan yang memiliki kolom 'node'.
        numeric_cols: Daftar nama kolom numerik yang akan di-capping. 
                      Default mengacu ke config.NUMERIC_COLS.
        num_nodes: Jumlah node. Default mengacu ke config.NUM_NODES.
    
    Returns:
        pd.DataFrame dengan nilai outlier yang sudah di-capping.
    """
    if numeric_cols is None:
        numeric_cols = NUMERIC_COLS
    if num_nodes is None:
        num_nodes = NUM_NODES

    node_frames = []
    for node_id in range(1, num_nodes + 1):
        df_node = df[df["node"] == node_id].copy()
        for col in numeric_cols:
            if col in df_node.columns:
                df_node[col] = _cap_outliers_iqr(df_node[col])
        node_frames.append(df_node)

    return pd.concat(node_frames, ignore_index=True)


# Feature Engineering - fitur waktu, lag/diff, rasio gas
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur turunan yang mereplikasi notebook 04_feature-engineering:
      - hour, minute, minute_of_day   (fitur temporal)
      - h2s_diff, so2_diff            (turunan berbasis lag)
      - gas_ratio_so2_h2s             (fitur rasio)
    
    Seluruh fitur dihitung per node.
    
    Args:
        df: DataFrame yang minimal memiliki kolom 'datetime', 'h2s', 'so2', dan 'node'.
    
    Returns:
        pd.DataFrame dengan kolom fitur baru yang telah ditambahkan.
    """
    df = df.copy()

    # Pastikan datetime memiliki tipe yang benar
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # --- Fitur temporal ---
    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # --- Fitur berbasis lag (dihitung per node) ---
    df["h2s_diff"] = df.groupby("node")["h2s"].diff()
    df["so2_diff"] = df.groupby("node")["so2"].diff()

    # --- Rasio gas ---
    df["gas_ratio_so2_h2s"] = df["so2"] / (df["h2s"] + 1e-6)

    # Mengisi nilai NaN dari diff dengan 0
    df["h2s_diff"] = df["h2s_diff"].fillna(0)
    df["so2_diff"] = df["so2_diff"].fillna(0)

    return df


# Full preprocessing pipeline
def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menjalankan keseluruhan pipeline preprocessing:
      1. Menangani outlier (IQR capping per node)
      2. Menerapkan feature engineering (fitur temporal, lag/diff, rasio gas)
    
    Args:
        df: DataFrame gabungan mentah dari proses pemuatan data.
    
    Returns:
        pd.DataFrame yang siap digunakan untuk pelatihan model.
    """
    df = handle_outliers(df)
    df = apply_feature_engineering(df)
    return df
