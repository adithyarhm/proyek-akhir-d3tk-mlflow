from src.config import *
from src.train import train_model
from src.data.data_loader import load_data

def main():
    df = load_data()

    lr_metrics, rfr_metrics, xgb_metrics = train_model(df)
    
    print(f"Metrics Linear Regression:")
    print(lr_metrics)

    print(f"Metrics Random Forest Regressor:")
    print(rfr_metrics)

    print(f"Metrics XGBoost:")
    print(xgb_metrics)
    

if __name__ == "__main__":
    main()