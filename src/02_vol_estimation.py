import os 
import numpy as np 
import pandas as pd 

DATA_DIR = "data"
BASE = pd.read_csv(os.path.join(DATA_DIR, "market_base.csv"), parse_dates=["Date"], index_col="Date")

ret2 = BASE["ret"] **2
BASE["RV21_hist"] = np.sqrt(252.0 * ret2.rolling(21, min_periods=21).sum())

lam = 0.94
s=np.nan; ew = []
for x in ret2:
    s = x if np.isnan(s) else lam * s + (1-lam)*x
    ew.append(s)
BASE["EWMA21"] = np.sqrt(252.0 * pd.Series(ew, index=BASE.index))

BASE.to_csv(os.path.join(DATA_DIR, "market_base.csv"))
