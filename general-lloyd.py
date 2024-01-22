import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 5000
mu = 0
sigma = 1
eps = 0.01
num_quantizers = 4

# Simulate the Binary Symmetric Channel
transition_matrix = np.matrix([[1 - eps, eps], [eps, 1 - eps]])

source_samples = np.sort(np.random.normal(mu, sigma, n))
centroids = np.linspace(-1, 1, num_quantizers)

# Function that calculates the distortion of each sample while accounting for
# channel noise (the sample may have been placed into the wrong bin).
def distortion_with_channel_noise(bins, centroids):
  distortion = 0
  for i in range(transition_matrix.shape[0]):
    for j in range(transition_matrix.shape[1]):
      for sample in bins[j]:
        distortion += transition_matrix[i,j]*(sample - centroids[i])**2
  return distortion

# Function that assigns each source sample to a bin while accounting for 
# channel noise using the NNC
def bin_assignment_with_channel_noise(samples, centroids):
  bin_dictionary = {} # dictionary where each key is the bin identifier (centroid) and the value is a list of all points assigned to that bin
  distortion_list = [] # list that will hold the distortion values calculated for each bin

  # Initialize each bin list 
  for i in range(transition_matrix.shape[0]):
    bin_dictionary[i] = [] 
    
  for sample in samples:
    for i in range(transition_matrix.shape(0)):
      distortion_for_bin = 0
      for j in range(transition_matrix.shape[1]):
        distortion_for_bin += transition_matrix[i,j]*(sample - centroids[j])**2
      distortion_list.append(distortion_for_bin)
      
    index = np.array(distortion_list).argmin
    bin_dictionary[index].append(sample)
  return bin_dictionary





  


