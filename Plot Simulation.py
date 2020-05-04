Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
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


