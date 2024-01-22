import numpy as np
import matplotlib.pyplot as plt

R = 1
N = 2**R

mu = 0
sigma = 1
eps = 0.01
num_samples = 5000

transition_matrix = [[1 - eps, eps], [eps, 1 - eps]]

# Function that calculates distortion for a bin and its designated centroid
def calc_distortion_for_bin(bin, centroid):
  distortion = 0
  for sample in bin:
    distortion += (sample - centroid)**2
  return distortion

def calc_distortion_for_all_bins(bins, centroids):
  distortion = 0
  for i in range(0, N): # loop over each bin
    for j in range(0, N): # loop over each centroid
      distortion += transition_matrix[i][j] * calc_distortion_for_bin(bins[i], centroids[j])
  return distortion / num_samples

# Function that assigns each sample to a bin using the Nearest Neighbor Condition
def assign_samples_to_bins(samples, centroids):
  bins = [[] for _ in range(N)]
    
  for sample in samples:
    distortion = -1
    index = 0 # bin to place a sample into
    for i in range(0, N):
      new_distortion = 0
      for j in range(0, N):
        new_distortion += transition_matrix[i][j] * (sample - centroids[j])**2
      if new_distortion < distortion or distortion == -1:
        distortion = new_distortion
        index = i
    bins[index].append(sample)
    
  return bins

# Function that re calculates the centroids based on the current bins and the transition matrix using the Centroid Condition
def calculate_centroids(bins):
  centroids = [[0] for _ in range(N)]
  
  for i in range(0, N):
    numerator = 0
    denominator = 0
    for j in range(0, N):
      numerator += transition_matrix[i][j] * sum(bins[j])
      denominator += transition_matrix[i][j] * len(bins[j])
    centroids[i] = numerator / denominator
    
  return centroids

def general_lloyds_algorithm(samples):
  distortion = []
  centroids = [-1, 4]
  
  bins = assign_samples_to_bins(samples, centroids)
  distortion.append(calc_distortion_for_all_bins(bins, centroids))
  
  i = 0
  while True:
    i += 1
    bins = assign_samples_to_bins(samples, centroids)
    centroids = calculate_centroids(bins)
    
    distortion.append(calc_distortion_for_all_bins(bins, centroids))
    if abs(distortion[i] - distortion[i-1]) / distortion[i-1] < 0.0001:
      break
    
  return [centroids, bins]

source_samples = np.random.normal(mu, sigma, num_samples)
[centroids, bins] = general_lloyds_algorithm(source_samples)

print(centroids)
print(bins)