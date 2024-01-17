import numpy as np
import matplotlib.pyplot as plt

def lloydSim(mu, sigma, n, cb_n, eps):
    # Init test data and codebook
    T = np.sort(np.random.normal(mu, sigma, n))

    cb_1 = np.linspace(-1, 1, cb_n)
    cb, D, m = llyodRecursive(T, cb_1, eps, 0, 1)
    return cb, D, m

def llyodRecursive(T, cb_m, eps, D_last, m):
    # Partition using NNC
    cb_m = np.sort(cb_m)
    R = [[] for _ in range(len(cb_m))]
    i = 0

    for d in range(len(T)):
        while True:
            if i == len(cb_m) - 1:
                R[i].append(T[d])
                break
            if abs(cb_m[i] - T[d]) < abs(cb_m[i + 1] - T[d]):
                R[i].append(T[d])
                break
            else:
                i += 1

    # If the first loop, calculate initial codebook distortion for D_last
    if m == 1:
        D_last = 0
        for i in range(len(cb_m)):
            D_last += sum(R[i])
        D_last /= len(T)

    # Find optimal codebook using partition R and CC
    cb = np.zeros(len(cb_m))
    for i in range(len(cb_m)):
        cb[i] = np.mean(R[i])

    # Check if change in distortion < eps (Optimal Codebook Found)
    D = 0
    for i in range(len(cb)):
        for d in range(len(R[i])):
            D += np.square(cb[i] - R[i][d])

    D /= len(T)

    m += 1

    if (D - D_last) / D >= eps:
        cb, D, m = llyodRecursive(T, cb, eps, D, m)

    return cb, D, m

# Parameters
n = 5000
mu = 0
sigma = 1
eps = 0.01

# Run simulations
cb1, D1, m1 = lloydSim(mu, sigma, n, 1, eps)
cb2, D2, m2 = lloydSim(mu, sigma, n, 2, eps)
cb4, D4, m4 = lloydSim(mu, sigma, n, 4, eps)
cb8, D8, m8 = lloydSim(mu, sigma, n, 8, eps)

# Plot distortion results
X = [1, 2, 4, 8]
Y = [D1, D2, D4, D8]
print(Y)
plt.figure()
plt.plot(X, Y)
plt.xlabel('Codebook Size (n)')
plt.ylabel('Average Distortion')
plt.title('Distortion for n-length Codebook')
plt.show()

# Output optimal codebooks
print('Optimal Codebooks:')
for i in range(len(X)):
    print(f'n={X[i]}:', cb1 if i == 0 else (cb2 if i == 1 else (cb4 if i == 2 else cb8)))
