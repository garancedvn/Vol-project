import os 
import pandas as pd 
import numpy as np 
import yfinance as yf 

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok = True)

#SPY dayly data abd VIX 
spy = yf.download("SPY", start="2010-01-01", progress=False)
vix = yf.download("^VIX", start="2010-01-01", progress= False)

spy = spy[["Close"]].rename(columns={"Close": "SPY_Close"})
vix = vix[["Close"]].rename(columns={"Close": "VIX_IV_30d"})

#Put all the data together and remove the dates that are not in both the VIX and SPY
df = spy.join(vix, how="inner")

#Get the log returns for each day
df["ret"] = np.log(df["SPY_Close"] / df["SPY_Close"].shift(1))

window = 21 #About 21 trading days in a month 
df["RV21_fwd"] = (
    np.sqrt(252) * df["ret"].shift(-1).rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)))
)

df.to_csv(os.path.join(DATA_DIR, "market_base.csv"))
print("Saved -> data/market_base.csv  (cols: SPY_Close, VIX_IV_30d, ret, RV21_fwd)")