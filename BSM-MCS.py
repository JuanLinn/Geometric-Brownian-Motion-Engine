import numpy as np
import pandas as pd
import scipy.stats as si
import pandas_datareader as pdr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

#a function to take the parameters given in the Black-Scholes model above that will return the option price (CC).
def bsm_call(S_t, K, r, sigma, T):
    den = 1 / (sigma * np.sqrt(T))
    d1 = den * (np.log(S_t / K) + (r + 0.5 * sigma ** 2) * T)
    d2 = den * (np.log(S_t / K) + (r - 0.5 * sigma ** 2) * T)
    C = S_t * stats.norm.cdf(d1) \
        - K * np.exp(-r * T) * stats.norm.cdf(d2)
    return C

#With the above function, we can calculate the theoretical price of a given European call option.
#Just because we can get the theoretical price, it doesn’t mean that’s actually what is available on the market.

#Another method to compute the bsm call option
def euro_vanilla_call(S, K, T, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    call = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))

    return call
#funtion for put
def euro_vanilla_put(S, K, T, r, sigma):
    # S: spot price
    # K: strike price
    # T: time to maturity
    # r: interest rate
    # sigma: volatility of underlying asset

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))

    return put
#Example for Facebook stock
start_date = '2018-01-01'
end_date = '2019-01-04'
fb_data = pdr.DataReader("AAPL", "yahoo", start_date, end_date)
fb_data.tail()
"""
we simply calculate the standard deviation of the daily returns. 
This gives us the daily volatility, but because our pricing model relies on annual volatility
we can multiply the daily volatility by the square root of 252 
(approximate number of trading days per year) to get the annual value.
"""
# Calculate volatility
# Returns
ret = fb_data['Close'][1:].values / fb_data['Close'][:-1].values - 1
# Volatility
sigma = np.std(ret) * np.sqrt(252)
print('sigma is ', sigma)
#This value is the annualized historical volatility.

# Get the risk-free rate
rfr = pdr.DataReader("DGS10", "fred", end_date)
rfr.head()

# Get the opening price on the day of interest
S_t = fb_data['Open'][-1]
print('S is ', S_t)
# Range of strike prices
K = S_t *(1 + np.linspace(0.05, 1, 20))
print("K is \n", K)
# Risk free rate on day of interest
r = rfr.loc[fb_data.index[-1]][0]
print("r is ", r)
# Time to maturity in year fractions
T = 0.5

# Calculate option prices
C = [bsm_call(S_t, k, r / 100, sigma, T) for k in K]
euro_call = euro_vanilla_call(S_t, K,T, r/100, sigma)
print("C is equal \n", C)
#print("euro_call is equal \n %d", euro_call)

plt.figure(figsize=(12,8))
plt.plot(K, euro_call)
plt.xlabel("Strike Price")
plt.ylabel("Option Price")
plt.title("Option Price vs. Strike Price for 6-Month European Call Options")
#plt.show()

#Monte Carlo Option Pricing Model Implementation

# Keep values from previous BSM for comparison
K_bsm = K[0]
C_bsm = euro_call[0]

np.random.seed(20)

# Parameters - same values as used in the example above
# repeated here for a reminder, change as you like
# Initial asset price
S_0 = S_t
# Strike price for call option
K = K_bsm
# Time to maturity in years
T = 0.5
# Risk-free rate of interest
r = rfr.loc[fb_data.index[-1]][0] / 100
print('r is ', r)
# Historical Volatility
sigma = np.std(ret) * np.sqrt(252)
print('sigma is ', sigma)
# Number of time steps for simulation
n_steps = int(T * 252)
# Time interval
dt = T / n_steps
# Number of simulations
N = 30000
# Zero array to store values (often faster than appending)
S = np.zeros((n_steps, N))
S[0] = S_0
print('S is ', S[0])
for t in range(1, n_steps):
    # Draw random values to simulate Brownian motion
    Z = np.random.standard_normal(N)
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * \
                             dt + (sigma * np.sqrt(dt) * Z))
# Sum and discount values
C = np.exp(-r * T) * 1 / N * np.sum(np.maximum(S[-1] - K, 0))
P = np.exp(-r * T) * 1 / N * np.sum(np.maximum(K - S[-1], 0))
CAON = []; PAON=[]
for i in range(1, N):
 if S[-1,i]>=K :
    CAON.append(S[-1,i])
 else:
    PAON.append(S[-1,i])
CA = np.exp(-r * T) * 1 / N * np.sum(CAON)
PA = np.exp(-r * T) * 1 / N * np.sum(PAON)

CCON = []; PCON = []
for i in range(1, N):
 if S[-1,i]>=K :
    CCON.append(K)
 else:
    PCON.append(K)
CC = np.exp(-r * T) * 1 / N * np.sum(CCON)
PC = np.exp(-r * T) * 1 / N * np.sum(PCON)
Fo = np.exp(-r * T) * 1 / N * np.sum(S[-1])



print("Strike price: {:.2f}".format(K_bsm))
print('difference beteern BSM and MC ', (C-C_bsm))
print("BSM Option Value Estimate: {:.2f}".format(C_bsm))
print("Monte Carlo Option Value Estimate: {:.2f}".format(C))
print("Monte Carlo CAON Option Value Estimate: {:.2f}".format(CA))
print("Monte Carlo PAON Option Value Estimate: {:.2f}".format(PA))
print("Monte Carlo CCON Option Value Estimate: {:.2f}".format(CC))
print("Monte Carlo PCON Option Value Estimate: {:.2f}".format(PC))
print("Monte Carlo Forward Option Value Estimate: {:.2f}".format(Fo))

plt.figure(figsize=(12,8))
plt.plot(S[:,0:1000])
plt.axhline(K, c="k", xmin=0,
            xmax=n_steps,
           label="Strike Price")
plt.xlim([0, n_steps])
plt.ylabel("Non-Discounted Value")
plt.xlabel("Time step")
plt.title("First 20 Option Paths")
plt.legend(loc="best")
plt.show()


