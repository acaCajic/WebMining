import pandas as pd
import numpy as np
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error

def zero_shot_forecast(csv_file: str, output_csv: str, min_points: int = 20):
    # Učitavanje CSV fajla
    df = pd.read_csv(csv_file)
    df["Created Date"] = pd.to_datetime(df["Created Date"])

    results = []

    # Konfiguracija modela TimesFM
    forecast_config = ForecastConfig(
        max_context=1024,                
        max_horizon=256,                 
        normalize_inputs=True,           
        window_size=0,
        per_core_batch_size=1,           
        use_continuous_quantile_head=False,
        force_flip_invariance=True,
        infer_is_positive=True,          
        fix_quantile_crossing=True,
        return_backcast=False,
    )

    # Inicijalizacija i kompajliranje modela
    model = TimesFM_2p5_200M_torch(forecast_config=forecast_config)
    model.compile(forecast_config)

    # Iteracija kroz sve artikle
    for item_no, df_item in df.groupby("Item_No"):
        # Sortiranje po datumu
        df_item = df_item.sort_values("Created Date")
        series = df_item["Quantity"].values

        # Ako je serija prekratka onda preskoci
        if len(series) < min_points:
            continue

        # Podela serije na trening i test (70/30)
        split_idx = int(len(series) * 0.7)
        train_series = series[:split_idx]
        test_series = series[split_idx:]
        horizon = len(test_series)

        # Predikcija sa modelom
        forecast = model.forecast(inputs=[train_series], horizon=horizon)
        y_pred = forecast[0][0]
        y_true = test_series

        # Racunanje gresaka
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Spremanje rezultata po danima samo za test podatke
        test_dates = df_item["Created Date"].iloc[split_idx:].values
        for date, true, pred in zip(test_dates, y_true, y_pred):
            results.append({
                "Item_No": item_no,
                "Date": date,
                "Actual": float(true),
                "Predicted": float(pred),
                "MAE_Item": round(mae, 2),
                "RMSE_Item": round(rmse, 2)
            })

    results_df = pd.DataFrame(results)
    # Cuvanje rezultata u CSV
    results_df.to_csv(output_csv, index=False)