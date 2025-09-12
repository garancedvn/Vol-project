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
    am = arch_model(r_train, vol="Garch", p=1, q=1, dist= "normal").fit()
    var1 = am.forecast(horizon=1).variance.values[-1,0] / (100**2)
    garch21 = np.sqrt(252 *21 * var1)
    forecast_series.iloc[t] = garch21

BASE["GARCH_RV21_forecast"] = forecast_series
BASE["IV_30d_proxy"] = BASE["VIX_IV_30d"] / 100.0

buffer = 0.02
BASE["signal"] = 0 
BASE.loc[BASE["GARCH_RV21_forecast"] > BASE["IV_30d_proxy"] + buffer, "signal"] = 1 
BASE.loc[BASE["GARCH_RV21_forecast"] < BASE["IV_30d_proxy"] - buffer, "signal"] = -1 
BASE["side"] = BASE["signal"]

# PnL proxy
scale = 1    
cost = 1
entries_enter_trade = BASE[(BASE["signal"] != 0) & (BASE["signal"].shift(1).fillna(0)==0)].copy()
trades = []

for dt in entries_enter_trade:
    pos = BASE.index.get_loc(dt)
    exit_pos = pos + 21

    if exit_pos >= len(BASE.index):
        continue 

    exit_dt = BASE.index[exit_pos]

    side = BASE.at[dt, "side "]
    iv_entry = BASE.at[dt, "IV_30d_proxy"]
    rv21_forecast = BASE.at[dt, "GARCH_RV21_forecast"]

    if pd.isna(side) or pd.isna(iv_entry) or pd.isna(rv21_forecast):
        continue

    pnl = ((rv21_forecast - iv_entry) * side * scale)

    trades.append({
        "entry_dt" : dt,
        "exit_dt": exit_dt,
        "side": side, 
        "iv_entry": iv_entry, 
        "rv_fwd": rv21_forecast,
        "pnl": pnl

    })

trades_df = pd.DataFrame(trades).set_index("entry_dt").sort_index()

