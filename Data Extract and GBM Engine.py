Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> #Example for Facebook stock
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