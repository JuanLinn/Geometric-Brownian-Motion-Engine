Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
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

