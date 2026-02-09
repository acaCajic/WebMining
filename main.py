from src.data_preprocessing import preprocess_sales_data
from src.zero_shot import zero_shot_forecast
from src.few_shot import few_shot_forecast
from src.visualization import plot_item_forecast

RAW_FILE = "data/DemoRetail22.xlsx"
PROCESSED_FILE = "data/timesfm_dataset.csv"
ZERO_SHOT_FILE = "data/timesfm_zero_shot.csv"
FEW_SHOT_FILE = "data/timesfm_few_shot.csv"

def main():
    preprocess_sales_data(RAW_FILE, PROCESSED_FILE)

    zero_shot_forecast(PROCESSED_FILE, ZERO_SHOT_FILE)

    few_shot_forecast(PROCESSED_FILE, FEW_SHOT_FILE)

    plot_item_forecast(ZERO_SHOT_FILE)
    plot_item_forecast(FEW_SHOT_FILE)

if __name__ == "__main__":
    main()