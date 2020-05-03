import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import requests 
URL = "https://fxmarketapi.com/apipandas"
params = {'currency' : 'EURUSD',
'start_date' : '2020-01-01',
'end_date':'2020-04-30',
'api_key':'YhWKRFUJou_ElFTtsqeg'}
response = requests.get("https://fxmarketapi.com/apipandas", params=params)
eur_usd= pd.read_json(response.text)
eur_usd.drop(eur_usd.tail(1).index,inplace=True)
hist = eur_usd['close']
print(eur_usd)
hist.plot(title="EUR - USD") 
S0 = hist[-1]
dt=1
T=20
N=T/dt
t_progression = np.arange(1, int(N)+1)
eur_usd['daily_return'] = (eur_usd['close']/ eur_usd['close'].shift(1)) -1
eur_usd.dropna(inplace = True)
mu = np.mean(eur_usd['daily_return'])
print(mu)
sigma = np.std(eur_usd['daily_return'])
print (sigma)
scen_size = 100
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
end_date = '2020-05-01'
pred_end_date = '2020-05-31'
plt.figure(figsize = (20,10))

for i in range(scen_size):
    plt.title("Daily Volatility: " + str(sigma))
    plt.plot(pd.date_range(start = end_date, 
                end = pred_end_date, freq = 'D').map(lambda x:
                x if x.isoweekday() in range(1, 6) else np.nan).dropna(), S[i, :])
    plt.ylabel('Stock Prices, $')
    plt.xlabel('Prediction Days')
    
plt.show()
