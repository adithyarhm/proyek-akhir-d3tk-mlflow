# ML Data Pipeline Implementation Summary

## Overview

The notebook-based ML pipeline from `/notebooks` has been modularized into reusable Python scripts within the `src/` directory. The pipeline follows this flow:

```mermaid
graph LR
    A["Raw Node CSVs<br/>(data/raw/)"] -->|load_and_merge_raw_nodes| B["Combined DataFrame"]
    B -->|handle_outliers| C["Outlier-Capped Data"]
    C -->|apply_feature_engineering| D["Feature-Engineered Data"]
    D -->|train_model| E["Trained Models + Metrics"]
```

## Files Created/Modified

### Modified Files

| File | Description |
|------|-------------|
| [config.py](file:///c:/Users/adith/Desktop/AI/proyek-akhir-d3tk-mlflow/src/config.py) | Centralized all configuration: data paths, column definitions, feature lists, hyperparameters |
| [data_loader.py](file:///c:/Users/adith/Desktop/AI/proyek-akhir-d3tk-mlflow/src/data/data_loader.py) | Rewrote with `load_and_merge_raw_nodes()` (replicates notebook 01) and `load_data()` |
| [train.py](file:///c:/Users/adith/Desktop/AI/proyek-akhir-d3tk-mlflow/src/train.py) | Updated feature/target separation using centralized `FEATURES` and `TARGET_COL` |
| [main.py](file:///c:/Users/adith/Desktop/AI/proyek-akhir-d3tk-mlflow/main.py) | Full pipeline orchestration: ingestion → preprocessing → training → evaluation |

### New Files

| File | Description |
|------|-------------|
| [preprocess.py](file:///c:/Users/adith/Desktop/AI/proyek-akhir-d3tk-mlflow/src/preprocess.py) | IQR-based outlier capping + feature engineering (time, diff, ratio features) |
| `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/evaluation/__init__.py` | Package init files for proper module discovery |

## Pipeline Steps Detail

### Step 1: Data Ingestion (`data_loader.py`)
- **Function:** `load_and_merge_raw_nodes()`
- Loads 6 individual node CSVs from `data/raw/`
- Adds spatial metadata (node ID, location, elevation, latitude, longitude)
- Casts datetime and numeric columns to proper types
- Merges into a single DataFrame

### Step 2: Preprocessing (`preprocess.py`)
- **Function:** `handle_outliers()` — IQR-based capping per node for `h2s`, `so2`, `hum`, `temp`, `windspeed`
- **Function:** `apply_feature_engineering()` — Creates:
  - `hour`, `minute`, `minute_of_day` (temporal features)
  - `h2s_diff`, `so2_diff` (lag-based derivatives per node)
  - `gas_ratio_so2_h2s` (ratio: `so2 / (h2s + 1e-6)`)
- **Function:** `preprocess_pipeline()` — Runs both steps in sequence

### Step 3: Training (`train.py`)
- **Function:** `train_model(df)`
- Splits data: 70% train, 15% validation, 15% test
- Trains 4 models: Linear Regression, Random Forest, XGBoost, SVR
- All use `MultiOutputRegressor` for multi-target prediction
- Returns models and final test metrics

### Step 4: Orchestration (`main.py`)
- Calls each step in sequence with clear logging
- Saves processed data to `data/processed/node_combined_final.csv`
- Prints all metrics

## Key Design Decisions

1. **Centralized config:** All paths, column names, and hyperparameters in `config.py`
2. **Per-node processing:** Outlier capping and lag features are computed per node, matching the notebook approach
3. **Separation of concerns:** Data loading, preprocessing, and training are independent modules
4. **Backward compatibility:** `load_data()` still exists for loading pre-processed CSVs
