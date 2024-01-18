import numpy as np
import matplotlib.pyplot as plt

def lloydSim(mu, sigma, n, num_quantizers, eps):
  # Init test data and codebook
  source_samples = np.sort(np.random.normal(mu, sigma, n))

  default_codebook = np.linspace(-1, 1, num_quantizers)
  cb, D, count = lloydRecursive(source_samples, default_codebook, eps, 0, 1)
  return cb, D, count

def lloydRecursive(source_samples, current_codebook, eps, D_last, count):
  # Partition using NNC
  current_codebook = np.sort(current_codebook)
  bins = [[] for _ in range(len(current_codebook))]
  i = 0

  for j in range(len(source_samples)):
    while True:
      if i == len(current_codebook) - 1:
        bins[i].append(source_samples[j])
        break
      if abs(current_codebook[i] - source_samples[j]) < abs(current_codebook[i + 1] - source_samples[j]):
        bins[i].append(source_samples[j])
        break
      else:
          i += 1

  # If the first loop, calculate initial codebook distortion for D_last
  if count == 1:
    D_last = 0
    for i in range(len(current_codebook)):
        D_last += sum(bins[i])
    D_last /= len(source_samples)

  # Find optimal codebook using partition R and CC
  new_codebook = np.zeros(len(current_codebook))
  for i in range(len(current_codebook)):
    new_codebook[i] = np.mean(bins[i])

  # Check if change in distortion < eps (Optimal Codebook Found)
  D = 0
  for i in range(len(new_codebook)):
    for j in range(len(bins[i])):
      bin_value = bins[i][j]
      D += np.square(new_codebook[i] - bin_value)

  D /= len(source_samples)

  count += 1

  if (D - D_last) / D >= eps:
    new_codebook, D, count = lloydRecursive(source_samples, new_codebook, eps, D, count)

  return new_codebook, D, count

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

# Output optimal codebooks
print('Optimal Codebooks:')
for i in range(len(X)):
  print(f'n={X[i]}:', cb1 if i == 0 else (cb2 if i == 1 else (cb4 if i == 2 else cb8)))
    
plt.show()
