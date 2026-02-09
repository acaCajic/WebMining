import pandas as pd
import numpy as np
from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Maksimalna duzina konteksta za predikciju
MAX_CONTEXT = 1024
# Maksimalni broj tacaka po seriji koje model gleda
MAX_POINTS_PER_SERIES = 128
# Minimalan broj tacaka serije
MIN_POINTS = 20
# Procenat test podataka
TEST_RATIO = 0.3

def split_train_test(series, test_ratio=TEST_RATIO):
    # Deli seriju na trening i test skup prema test_ratio  
    split = int(len(series) * (1 - test_ratio))
    return series[:split], series[split:]

def prepare_train_context(train_series, series):
    # Proverava da serija podataka za jedan element nije manji od MIN_POINTS
    if len(series) < MIN_POINTS:
        return None
    # Skracuje trening seriju na maksimalan broj tacaka koje model moze primiti za jedan artikal
    return train_series[-MAX_POINTS_PER_SERIES:]

def build_collections(series_list):
    # Sadrzi sve kolekcije podataka
    collections = []
    # Popunjava se do MAX_CONTEXT podataka a zatim se ta vrednost prebacuje
    # u collection a vrednost current_collection se postavlja na prazan niz
    current_collection = []
    # Proverava da current_collection ne predje MAX_CONTEXT
    current_tokens = 0

    for s in series_list:
        # Racuna duzinu trening serije
        length = len(s["train"])
        # Proverava da maksimalan broj current_collection ne predje MAX_CONTEXT
        if current_tokens + length > MAX_CONTEXT:
            collections.append(current_collection)
            current_collection = []
            current_tokens = 0
        # Popunjava se current_collection dok ne dodje do MAX_CONTEXT
        current_collection.append(s)
        current_tokens += length

    # Za poslednju kolekciju podataka
    if current_collection:
        collections.append(current_collection)
    return collections

def few_shot_forecast(csv_file: str, output_csv: str):
    # Ucitavanje podataka
    df = pd.read_csv(csv_file)
    df["Created Date"] = pd.to_datetime(df["Created Date"])

    all_series = []

    # Priprema svih serija u obliku koji ce nam biti potreban za pravljanje
    # kolekcija na osnovu kojih se vrsi predikcija
    for item in df["Item_No"].unique():
        df_item = df[df["Item_No"] == item].sort_values("Created Date")
        s = df_item["Quantity"].values
        dates = df_item["Created Date"].values

        # Dobijamo vrednosti koje se koriste za trening i kasnije za proveru
        train, test = split_train_test(s)
        _, test_dates = split_train_test(dates)

        # Dobijamo maksimalan broj podataka o prodaji datog artikla koji je 
        # moguce da udje u kolekciju za taj odredjeni artikal 
        train_ctx = prepare_train_context(train, s)
        if train_ctx is None or len(test) == 0:
            continue

        all_series.append({
            "item": item,
            "train": train_ctx,
            "test": test,
            "test_dates": test_dates
        })

    # Konfiguracija modela
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

    results = []

    # Iteracija kroz kolekcije
    for collection in build_collections(all_series):
        inputs = [b["train"] for b in collection]
        clean_inputs = [arr.astype(int).tolist() for arr in inputs]
        # Za broj predvidjanja se uzima maksimalan broj iz liste test koji
        # jedan element sadrzi u datoj kolekciji
        horizon = max(len(b["test"]) for b in collection)

        forecast = model.forecast(inputs=clean_inputs, horizon=horizon)

        for i, b in enumerate(collection):
            item = b["item"]
            y_true = b["test"]
            # Posto se za broj predvidjanja uzima maksimalna duzina za test listu
            # koji jedan artikal u kolekciji sadrzi uzimamo duzinu podataka koliko
            # ima konkretni artikal u nizu test
            y_pred = forecast[0][i][:len(y_true)]

            # Racunanje gresaka
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            # Spremanje predikcija po danima
            for date, true, pred in zip(b["test_dates"], y_true, y_pred):
                results.append({
                    "Item_No": item,
                    "Date": pd.to_datetime(date),
                    "Actual": float(true),
                    "Predicted": float(pred),
                    "MAE_Item": round(mae, 2),
                    "RMSE_Item": round(rmse, 2)
                })

    results_df = pd.DataFrame(results)
    # Cuvanje predikcija u CSV
    results_df.to_csv(output_csv, index=False)