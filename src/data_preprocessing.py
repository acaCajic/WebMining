import pandas as pd
import openpyxl

def preprocess_sales_data(input_excel: str, output_csv: str):
    # Ucitavanje Excel fajla
    df = pd.read_excel(input_excel)

    # Normalizacija datuma
    df["Created Date"] = pd.to_datetime(df["Created Date"]).dt.normalize()

    df["Entry_Type"] = "Sale"

    # Pretvaranje kolicina u apsolutne vrednosti
    df["Quantity"] = df["Quantity"].abs()

    # Grupisanje po artiklu i datumu i sumiranje kolicina
    df_grouped = (
        df.groupby(["Item_No", "Created Date"], as_index=False)["Quantity"]
        .sum()
    )

    # Cuvanje pripremljenog CSV fajla
    df_grouped.to_csv(output_csv, index=False)