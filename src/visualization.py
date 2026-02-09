import pandas as pd
import matplotlib.pyplot as plt

def plot_item_forecast(csv_file: str, item_no: str = None): 
    results_df = pd.read_csv(csv_file)
    results_df["Date"] = pd.to_datetime(results_df["Date"])

    # Ako se ne prosledi item_no automatski se bira artikl sa najvise podataka
    if item_no is None:
        # Izbor artikla sa najvise podataka
        item_no = results_df.groupby("Item_No").size().sort_values(ascending=False).index[0]

    item_df = results_df[results_df["Item_No"] == item_no].sort_values("Date")

    plt.figure(figsize=(10,5))
    plt.plot(item_df["Date"], item_df["Actual"], linewidth=2, marker="o")
    plt.plot(item_df["Date"], item_df["Predicted"], linestyle="--", marker="o")
    plt.title(f"Actual vs Predicted for Item {item_no}")
    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.legend(["Actual", "Predicted"])
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()