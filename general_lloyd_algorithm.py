import numpy as np
import matplotlib.pyplot as plt
import math

def code_rate(codebook_length):
  return math.ceil(math.log2(codebook_length))

# Function that calculates the hamming distance between x and y.
# Will return the number of bits that differ between x and y
def hamming_distance(x, y):
  xor = x ^ y
  count = 0
  while xor:
    count += xor & 1
    xor >>= 1
  return count

def conditional_probability(i, j, error_probability, code_rate):
  dH = hamming_distance(i, j)
  return ((error_probability ** dH)) * ((1 - error_probability) ** (code_rate - dH))

# Function that calculates distortion across all bins
def calc_distortion_for_all_bins(bins, centroids, codebook_length, num_samples, channel_error_probability):
  distortion = 0
  
  for i in range(0, codebook_length): # loop over each bin
    for sample in bins[i]: # loop over each sample in bin
      for j in range(codebook_length):
        distortion += conditional_probability(i, j, channel_error_probability, code_rate(codebook_length)) * (sample - centroids[j])**2
        
  return distortion / num_samples

# Function that assigns each sample to a bin using the Nearest Neighbor Condition
def assign_samples_to_bins(samples, centroids, codebook_length, channel_error_probability):
  bins = [[] for _ in range(codebook_length)]
    
  for sample in samples:
    distortion = -1
    index = 0 # bin to place a sample into
    for i in range(0, codebook_length):
      new_distortion = 0
      for j in range(0, codebook_length):
        new_distortion += conditional_probability(i, j, channel_error_probability, code_rate(codebook_length)) * (sample - centroids[j])**2
      if new_distortion < distortion or distortion == -1:
        distortion = new_distortion
        index = i
    bins[index].append(sample)
    
  return bins

# Function that re calculates the centroids based on the current bins and the transition matrix using the Centroid Condition
def calculate_centroids(bins, codebook_length, channel_error_probability, num_samples):
  centroids = [0] * codebook_length
  
  for j in range(codebook_length):
    numerator = 0
    denominator = 0
    for i in range(codebook_length):
      numerator += conditional_probability(i, j, channel_error_probability, code_rate(codebook_length)) * (sum(bins[i]))
      denominator += conditional_probability(i, j, channel_error_probability, code_rate(codebook_length)) * (len(bins[i]))
    centroids[j] = numerator / denominator
    #added so that if the denominator == 0, the centroid isn't NAN
    if denominator==0:
      centroids[j] = 0    
  return centroids

def general_lloyds_algorithm(samples, num_samples, channel_error_probability, codebook_length):
  distortion = []
  centroids = np.linspace(-1, 1, codebook_length)
  
  bins = assign_samples_to_bins(samples, centroids, codebook_length, channel_error_probability)
  distortion.append(calc_distortion_for_all_bins(bins, centroids, codebook_length, num_samples, channel_error_probability))
  
  i = 0
  while True:
    i += 1
    bins = assign_samples_to_bins(samples, centroids, codebook_length, channel_error_probability)
    centroids = calculate_centroids(bins, codebook_length, channel_error_probability, num_samples)
    distortion.append(calc_distortion_for_all_bins(bins, centroids, codebook_length, num_samples, channel_error_probability))
    print(distortion)
    if abs(distortion[i] - distortion[i-1]) / distortion[i-1] <= channel_error_probability:
      break
  #returning the last distortion value (final iterated distortion)
  return [centroids, bins, distortion[-1]]


'''
def main():
  mu = 0
  sigma = 1
  epsilon = 0.01
  num_samples = 10**3
  channel_error_probability = 0
  normal_source_samples = np.random.normal(mu, sigma, num_samples)
  codebook_length = [1,2,4,8]
  
  distortion = [0,0,0,0]
  for i in range(len(distortion)):
    [centroids, bins, distortion[i]] = general_lloyds_algorithm(normal_source_samples, num_samples, channel_error_probability, epsilon, codebook_length[i])


  plt.figure()
  plt.plot([1,2,4,8], distortion)
  plt.xlabel('Iteration')
  plt.ylabel('Average Distortion')
  plt.title('Distortion for n-length Codebook')
  plt.show()

if __name__ == "__main__":
  main()
'''