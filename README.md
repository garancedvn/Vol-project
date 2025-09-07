# Volatility Trading Strategy: IV vs RV using GARCH

## Project Overview
This project explores a simple **systematic volatility trading strategy**
The idea is to compare IV from options to a forecast of RV obtained with a GARCH(1,1) model 
- If forecasted RV > IV then buy straddle (long valatility)
- If forecasted RV< IV then sell straddle (short volatility)
We then backtest this daily over history to evaluate profitability after transaction costs. 

## Data 
- **Underlying:** Daily OHLC prices for SPY, downloaded via `yfinance`.  
- **Options:** One near-30-day to expiry **ATM call + ATM put** per day to construct a straddle and extract implied volatility.  
  - If IV is available directly in the dataset → use it.  
  - Otherwise → invert the **Black–Scholes model** from option mid-prices.  

## Data Notes 
- Skip days with missing quotes.  
- Use liquid contracts only (front month, ATM).  
- Use **mid-prices** (average of bid and ask) and assume a fixed cost (e.g. $1 per contract each side) to make results realistic.

## Methodology 
1. Collect SPY daily returns and option prices.  
2. Estimate daily volatility forecasts with GARCH(1,1).  
3. Compare forecasted RV to market IV.  
4. Generate a trading signal (long/short straddle).  
5. Backtest strategy PnL including costs.  

## Evaluation 
- Forecast accuracy: correlation and error metrics (RMSE/MAE) of GARCH vs future RV.  
- Strategy performance: Sharpe ratio, max drawdown, win rate, and equity curve.  

## Tech Stack 
- Python (pandas, numpy, matplotlib, yfinance, arch, scipy)  
- Git + GitHub for version control  
- Jupyter Notebook for analysis and plots  

## Disclaimer 
This project is for **educational purposes only** and does not constitute financial advice.