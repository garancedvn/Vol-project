import os
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

DATA_DIR = "data"
BASE = pd.read_csv(os.path.join(DATA_DIR, "market_base.csv"), parse_dates=["Date"], index_col="Date").dropna()

rets = BASE["ret"].dropna()
rolling_window = 750
forecast_series = pd.Series(index=rets.index, dtype=float)

for t in range(rolling_window, len(rets) - 21):
    r_train = rets.iloc[:t]*100
    am = arch_model(r_train, vol="Garch", p=1, q=1, dist= "normal")
    var1 = am.forecast(horizon=1).variance.values[-1,0] / (100**2)
    garch21 = np.sqrt(252 *21 * var1)
    forecast_series.iloc[t] = garch21

BASE["GARCH_RV21_forecast"] = forecast_series
BASE["IV_30d_proxy"] = BASE["VIX_IV_30d"] / 100.0

buffer = 0.02
BASE["signal"] = 0 
BASE.loc[BASE["GARCH_RV21_forecast"] > BASE["IV_30d_proxy"] + buffer, "signal"] = 1 
BASE.loc[BASE["GARCH_RV21_forecast"] < BASE["IV_30d_proxy"] - buffer, "signal"] = -1 