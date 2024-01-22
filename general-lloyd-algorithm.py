import numpy as np
import matplotlib.pyplot as plt

# Function that calculates the hamming distance between x and y.
# Will return the number of bits that differ between x and y
def hamming_distance(x, y):
  xor = x ^ y
  count = 0
  while xor:
    count += xor & 1
    xor >>= 1
  return count

def conditional_probability(i, j, error_probability, N):
  dH = hamming_distance(i, j)
  return (error_probability ** dH) * (1 - error_probability) ** (N - error_probability)

# Function that calculates distortion across all bins
def calc_distortion_for_all_bins(bins, centroids, codebook_length, num_samples, eps):
  distortion = 0
  
  for i in range(0, codebook_length): # loop over each bin
    for sample in bins[i]: # loop over each sample in bin
      for j in range(codebook_length): 
        distortion += conditional_probability(i, j, eps, codebook_length) * (sample - centroids[j])**2
        
  return distortion / num_samples

# Function that assigns each sample to a bin using the Nearest Neighbor Condition
def assign_samples_to_bins(samples, centroids, codebook_length, eps):
  bins = [[] for _ in range(codebook_length)]
    
  for sample in samples:
    distortion = -1
    index = 0 # bin to place a sample into
    for i in range(0, codebook_length):
      new_distortion = 0
      for j in range(0, codebook_length):
        new_distortion += conditional_probability(i, j, eps, codebook_length) * (sample - centroids[j])**2
      if new_distortion < distortion or distortion == -1:
        distortion = new_distortion
        index = i
    bins[index].append(sample)
    
  return bins

# Function that re calculates the centroids based on the current bins and the transition matrix using the Centroid Condition
def calculate_centroids(bins, codebook_length, eps):
  centroids = [[0] for _ in range(codebook_length)]
  
  for i in range(0, codebook_length):
    numerator = 0
    denominator = 0
    for j in range(0, codebook_length):
      numerator += conditional_probability(i, j, eps, codebook_length) * sum(bins[j])
      denominator += conditional_probability(i, j, eps, codebook_length) * len(bins[j])
    centroids[i] = numerator / denominator
    
  return centroids

def general_lloyds_algorithm(samples, num_samples, eps):
  distortion = []
  centroids = [-1, 4]
  codebook_length = len(centroids)
  
  bins = assign_samples_to_bins(samples, centroids, codebook_length, eps)
  distortion.append(calc_distortion_for_all_bins(bins, centroids, codebook_length, num_samples, eps))
  
  i = 0
  while True:
    i += 1
    bins = assign_samples_to_bins(samples, centroids, codebook_length, eps)
    centroids = calculate_centroids(bins, codebook_length, eps)
    
    distortion.append(calc_distortion_for_all_bins(bins, centroids, codebook_length, num_samples, eps))
    if abs(distortion[i] - distortion[i-1]) / distortion[i-1] < 0.0001:
      break
    
  return [centroids, bins]

def main():
  mu = 0
  sigma = 1
  eps = 0.2
  num_samples = 5
  source_samples = np.random.normal(mu, sigma, num_samples)
  [centroids, bins] = general_lloyds_algorithm(source_samples, num_samples, eps)
  print(bins)
  print(centroids)

if __name__ == "__main__":
  main()
