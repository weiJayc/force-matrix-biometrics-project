import pandas as pd
from pathlib import Path

folder = Path("../dataset/jay")


df = pd.read_csv("../dataset/jay/20260703_152250_724535.csv")

for i in range(16):
    print(
        f"value_{i}",
        df[f"value_{i}"].unique()[:20]
    )