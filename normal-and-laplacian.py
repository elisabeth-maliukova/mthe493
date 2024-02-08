from general_lloyd_algorithm import general_lloyds_algorithm
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_normal_samples(rate, channel_error_probability):
  mu = 0
  sigma = 1
  epsilon = 0.01
  num_samples = 10**3
  
  normal_source_samples = np.random.normal(mu, sigma, num_samples)
  codebook_length = []
  distortion = []
  for i in range(rate):
    codebook_length.append(2**i)
    distortion.append(0)
  plt.figure()
  plt.xlabel('Codebook Size (n)')
  plt.ylabel('Distortion')
  plt.title('Distortion for n-length Codebook (Normal)')
  for error in channel_error_probability:
    for j in range(rate):
      [centroids, bins, distortion[j]] = general_lloyds_algorithm(normal_source_samples, num_samples, error, codebook_length[j])
    plt.plot(codebook_length, distortion)
  plt.legend([str(channel_error_probability[0]),str(channel_error_probability[1]),str(channel_error_probability[2]),str(channel_error_probability[3])])
  
def plot_laplacian_samples(rate, channel_error_probability):
  mean = 0
  scale = 1 / math.sqrt(2)
  epsilon = 0.01
  num_samples = 10**3
  
  laplacian_source_samples = np.random.laplace(mean, scale, num_samples)
  
  codebook_length = []
  distortion = []
  for i in range(rate):
    codebook_length.append(2**i)
    distortion.append(0)
  plt.figure()
  plt.xlabel('Codebook Size (n)')
  plt.ylabel('Distortion')
  plt.title('Distortion for n-length Codebook (Laplacian)')
  for error in channel_error_probability:
    for j in range(rate):
      [centroids, bins, distortion[j]] = general_lloyds_algorithm(laplacian_source_samples, num_samples, error, epsilon, codebook_length[j])
    plt.plot(codebook_length, distortion)
  plt.legend([str(channel_error_probability[0]),str(channel_error_probability[1]),str(channel_error_probability[2]),str(channel_error_probability[3])])
  
  
def main():
  plt.close('all')
  plot_normal_samples(4, [0, 0.01, 0.1, 0.5])
  plot_laplacian_samples(4, [0, 0.01, 0.1, 0.5])
  plt.show()
  
if __name__ == "__main__":
  main()