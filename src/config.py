## Konfigurasi Pipeline


# Data Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
RAW_NODE_FILES = [
    "node1.csv", "node2.csv", "node3.csv",
    "node4.csv", "node5.csv", "node6.csv",
]
NODE_COMBINED_RAW = "data/raw/node_combined.csv"
NODE_COMBINED_PREPROCESS = "data/processed/node_combined_preprocess.csv"
NODE_COMBINED_FINAL = "data/processed/node_combined_final.csv"

DATA_PATH = NODE_COMBINED_FINAL

# Model Saving
MODEL_SAVE_DIR = "saved-models"

# Column Definitions
NUMERIC_COLS = ["h2s", "so2", "hum", "temp", "windspeed"]
TARGET_COL = ["h2s", "so2"]

# Features used for training (after feature engineering)
FEATURES = [
    "h2s", "so2", "hum", "temp", "windspeed",
    "hour", "minute", "minute_of_day",
    "h2s_diff", "so2_diff", "gas_ratio_so2_h2s",
]

# Train/Test Split
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Number of Nodes
NUM_NODES = 6