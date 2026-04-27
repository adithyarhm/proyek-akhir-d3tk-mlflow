"""
DATA LOADER
=================
Menangani proses pemuatan file CSV node mentah, menggabungkannya menjadi satu DataFrame,
serta memuat data yang telah diproses sebelumnya.
"""

import pandas as pd
import os
from src.config import (
    RAW_DATA_DIR, RAW_NODE_FILES, NODE_COMBINED_RAW, DATA_PATH
)


# Pemetaan nama kolom (CSV raw -> nama standar)
# CSV node raw menggunakan:    Node, Location, Weather, DateTime, H2S, SO2,
#                              Humidity, Temperature, WindsSpeed
# Pipeline menggunakan:        node, location, weather, datetime, h2s, so2,
#                              hum, temp, windspeed
COLUMN_RENAME_MAP = {
    "Node": "node",
    "Location": "location",
    "Weather": "weather",
    "DateTime": "datetime",
    "H2S": "h2s",
    "SO2": "so2",
    "Humidity": "hum",
    "Temperature": "temp",
    "WindsSpeed": "windspeed",
}


# Metadata spasial untuk setiap node (dari notebook 01_load_audit)
NODE_METADATA = {
    1: {"location": "Dekat uap panas",
        "elevation": 2101.0,
        "latitude": -7.16687,
        "longitude": 107.401387
    },
    2: {"location": "Dekat sumber mata air",
        "elevation": 2195.0,
        "latitude": -7.167397,
        "longitude": 107.401775
    },
    3: {"location": "Hutan Mati",
        "elevation": 2196.0,
        "latitude": -7.167415,
        "longitude": 107.402914
    },
    4: {"location": "Area Pengunjung",
        "elevation": 2193.0,
        "latitude": -7.166614,
        "longitude": 107.403483
    },
    5: {"location": "Goa",
        "elevation": 2194.0,
        "latitude": -7.166418,
        "longitude": 107.4041
    },
    6: {"location": "Tangga Masuk Pengunjung",
        "elevation": 2200.0,
        "latitude": -7.166833,
        "longitude": 107.40411
    },
}


def load_and_merge_raw_nodes() -> pd.DataFrame:
    """
    Load file CSV tiap node dari direktori data raw,
    mengganti nama kolom menjadi format standar huruf kecil,
    menambahkan metadata spasial (elevation, lat, long),
    dan menggabungkannya menjadi satu DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame gabungan yang berisi seluruh data node.
    """
    frames = []
    for i, filename in enumerate(RAW_NODE_FILES, start=1):
        filepath = os.path.join(RAW_DATA_DIR, filename)
        df_node = pd.read_csv(filepath)
        
        # Ganti nama raw columns untuk standardized nama
        df_node = df_node.rename(columns=COLUMN_RENAME_MAP)
        
        # Tambahkan metadata spasial (node & location sudah ada dari CSV)
        meta = NODE_METADATA[i]
        df_node["elevation"] = meta["elevation"]
        df_node["latitude"] = meta["latitude"]
        df_node["longitude"] = meta["longitude"]
        
        frames.append(df_node)
    
    df_combined = pd.concat(frames, ignore_index=True)
    
    # Parsing kolom datetime
    if "datetime" in df_combined.columns:
        df_combined["datetime"] = pd.to_datetime(
            df_combined["datetime"], errors="coerce"
        )
    
    # Ubah kolom numerik menjadi float
    for col in ["h2s", "so2", "hum", "temp", "windspeed",
                "elevation", "latitude", "longitude"]:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(
                df_combined[col], errors="coerce"
            )
    
    return df_combined


def load_data() -> pd.DataFrame:
    """
    Memuat dataset akhir yang telah diproses (node_combined_final.csv).
    Dataset ini sudah siap digunakan untuk pelatihan setelah seluruh tahap
    preprocessing dan feature engineering diterapkan.
    
    Returns:
        pd.DataFrame: DataFrame akhir yang telah diproses.
    """
    df = pd.read_csv(DATA_PATH)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df