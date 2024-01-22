from general_lloyd_algorithm import general_lloyds_algorithm
import numpy as np
import matplotlib.pyplot as plt


  
def plot_normal_samples(rate, channel_error_probability):
  mu = 0
  sigma = 1
  epsilon = 0.01
  num_samples = 10**3
  
  normal_source_samples = np.random.normal(mu, sigma, num_samples)
  
  codebook_length = 2**rate
  [centroids, bins, distortion] = general_lloyds_algorithm(normal_source_samples, num_samples, channel_error_probability, epsilon, codebook_length)
  indexes = range(len(distortion))
  
  plt.figure()
  plt.plot(indexes, distortion)
  plt.xlabel('Codebook Size (n)')
  plt.ylabel('Average Distortion')
  plt.title('Distortion for n-length Codebook (Normal)')
  plt.show()
  
def plot_laplacian_samples(rate, channel_error_probability):
  mu = 0
  sigma = 1
  epsilon = 0.01
  num_samples = 10**3
  
  laplacian_source_samples = np.random.laplace(mu, sigma, num_samples)
  
  codebook_length = 2**rate
  [centroids, bins, distortion] = general_lloyds_algorithm(laplacian_source_samples, num_samples, channel_error_probability, epsilon, codebook_length)
  indexes = range(len(distortion))
  
  plt.figure()
  plt.plot(indexes, distortion)
  plt.xlabel('Codebook Size (n)')
  plt.ylabel('Average Distortion')
  plt.title('Distortion for n-length Codebook (Laplacian)')
  plt.show()
  
  
def main():
  plot_normal_samples(3, 0.1)
  plot_laplacian_samples(3, 0.1)
  
if __name__ == "__main__":
  main()