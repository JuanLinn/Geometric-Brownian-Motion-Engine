Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 23:03:10) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Sum and discount values
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