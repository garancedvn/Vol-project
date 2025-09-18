import os
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

def sharpe(daily):
    s = daily.std(ddof=1)
    return np.nan if (s == 0 or np.isnan(s)) else np.sqrt(252) * daily.mean() / s


DATA_DIR = "data"
BASE = pd.read_csv(os.path.join(DATA_DIR, "market_base.csv"), 
                   skiprows = 2,
                   parse_dates=["Date"], 
                   index_col="Date")

print(BASE.columns)

rets = BASE["ret"].dropna()
rolling_window = 750
forecast_series = pd.Series(index=rets.index, dtype=float)

for t in range(rolling_window, len(rets) - 21):
    r_train = rets.iloc[:t]*100
    am = arch_model(r_train, vol="Garch", p=1, q=1, dist= "normal").fit(disp="off")
    var1 = am.forecast(horizon=1).variance.values[-1,0] / (100.0**2)
    garch21 = np.sqrt(252 *21 * var1)
    forecast_series.iloc[t] = garch21

BASE["GARCH_RV21_forecast"] = forecast_series
BASE["IV_30d_proxy"] = BASE["VIX_IV_30d"] / 100.0

need = ["GARCH_RV21_forecast", "IV_30d_proxy", "RV21_fwd"]
BASE = BASE.dropna(subset=[c for c in need if c in BASE.columns])

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

for dt in entries_enter_trade.index:
    pos = BASE.index.get_loc(dt)
    exit_pos = pos + 21

    if exit_pos >= len(BASE.index):
        continue 

    exit_dt = BASE.index[exit_pos]

    side = BASE.at[dt, "side"]
    iv_entry = BASE.at[dt, "IV_30d_proxy"]
    rv_fwd = BASE.at[dt, "RV21_fwd"]

    if pd.isna(side) or pd.isna(iv_entry) or pd.isna(rv_fwd):
        continue

    pnl = ((rv_fwd - iv_entry) * side * scale) - cost

    trades.append({
        "entry_dt" : dt,
        "exit_dt": exit_dt,
        "side": side, 
        "iv_entry": iv_entry, 
        "rv_fwd": rv_fwd,
        "pnl": pnl

    })

trades_df = pd.DataFrame(trades).set_index("entry_dt").sort_index()
daily_pnl = pd.Series(0.0, index=BASE.index)

for c, r in trades_df.iterrows():
    if r["exit_dt"] in daily_pnl.index:
        daily_pnl.at[r["exit_dt"]] += r["pnl"]

if len(trades_df) > 0:
    start_idx = trades_df.index.min()
    daily_pnl = daily_pnl.loc[start_idx:]

eq_curve = daily_pnl.cumsum()
dd = eq_curve - eq_curve.cummax()

metrics = {
    "n_trades": int(len(trades_df)),
    "hit_rate": float((trades_df["pnl"] > 0).mean()) if len(trades_df) else np.nan,
    "avg_pnl": float(trades_df["pnl"].mean()) if len(trades_df) else np.nan,
    "total_pnl": float(trades_df["pnl"].sum()) if len(trades_df) else np.nan,
    "sharpe": float(sharpe(daily_pnl)),
    "max_dd": float(dd.min()) if len(eq_curve) else np.nan,
}

print(metrics)

