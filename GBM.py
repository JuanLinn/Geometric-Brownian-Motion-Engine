import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
start_date = '2020-03-01'
end_date = '2020-03-31'
pred_start_date = '2020-04-01'
pred_end_date = '2020-04-30'
msft = yf.Ticker("MSFT")
hist = msft.history(period='1d',start = '2020-3-1',end = '2020-3-31')
hist['Close'].plot(title="MSFT's stock price") 
S_hist = hist['Close']
S0 = S_hist[-1]
dt=1
days = pd.date_range(start = pd.to_datetime(end_date, 
              format = "%Y-%m-%d") + pd.Timedelta('1 days'), 
              end = pd.to_datetime(pred_end_date, 
              format = "%Y-%m-%d")).to_series(
              ).map(lambda x: 
              1 if x.isoweekday() in range(1,6) else 0).sum()
T = days
N=T/dt
t_progression = np.arange(1, int(N)+1)
hist['daily_return'] = (hist['Close']/ hist['Close'].shift(1)) -1
hist.dropna(inplace = True)
mu = np.mean(hist['daily_return'])
sigma = np.std(hist['daily_return'])
scen_size = 10
b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
print(b)
W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}
print(W)
drift = (mu - 0.5 * sigma**2) * t_progression
print("drift:\n", drift)
diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}
print("diffusion:\n", diffusion)
S = np.array([S0 * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)]) 
S = np.hstack((np.array([[S0] for scen in range(scen_size)]), S))
print(S)
plt.figure(figsize = (20,10))

for i in range(scen_size):
    plt.title("Daily Volatility: " + str(sigma))
    plt.plot(pd.date_range(start = end_date, 
                end = pred_end_date, freq = 'D').map(lambda x:
                x if x.isoweekday() in range(1, 6) else np.nan).dropna(), S[i, :])
    plt.ylabel('Stock Prices, $')
    plt.xlabel('Prediction Days')
    
plt.show()