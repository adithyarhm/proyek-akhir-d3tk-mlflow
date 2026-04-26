from src.config import *
from src.train import train_model
from src.data.data_loader import load_data
from src.save_models import save_trained_models

def main():
    df = load_data()

    models, metrics = train_model(df)
    lr, rfr, xgbr, svr = models
    lr_final_metrics, rfr_final_metrics, xgbr_final_metrics, svr_final_metrics = metrics

    # Panggil fungsi save models dari main.py
    #save_trained_models(lr, rfr, xgbr, svr)
    print("\n--- METRICS ---")

    # print(f"Metrics Linear Regression:\n", lr_final_metrics)

    print(f"Metrics Random Forest Regressor:\n", rfr_final_metrics)
    
    print(f"Metrics XGBoost:\n", xgbr_final_metrics)

    print(f"Metrics SVR:\n", svr_final_metrics)


if __name__ == "__main__":
    main()