import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm
import matplotlib.pyplot as plt

INVALID_CENTROID = -999999

def mod2_addition(a, b):
    return (a + b) % 2

def send_through_polya_channel(epsilon, delta, original_bins, centroids):
  num_centroids = len(centroids)
  
  # Calculate the number of bits needed to represent the centroids
  num_bits = int(np.ceil(np.log2(num_centroids)))

  # Generate the binary strings for each centroid
  binary_strings = [format(i, f'0{num_bits}b') for i in range(num_centroids)]
  
  # Associate each sample with its encoding directly
  sample_encoding_pairs = []
  for bin_index, bin in enumerate(original_bins):
    for sample in bin:
      sample_encoding_pairs.append((sample, binary_strings[bin_index]))
      
  distorted_pairs = simulate_polya_channel(epsilon, delta, sample_encoding_pairs)


  return distorted_pairs
      
  
def get_transition_prob(prev_state, epsilon, delta):
  return (epsilon + (prev_state * delta)) / (1 + delta)
  
  
def simulate_polya_channel(epsilon, delta, sample_encoding_pairs):
  distorted_pairs = []
  
  for sample, encoding in sample_encoding_pairs:
    distorted_encoding = [0] * len(encoding)
    z_process = [0] * len(encoding)
    z_process[0] = 1 if (np.random.rand() < epsilon) else 0
    
    for index, bit in enumerate(encoding):
      if (index == 0):
        distorted_encoding[0] = mod2_addition(int(bit), z_process[0])
      else:
        z_process[index] = 1 if np.random.rand() < get_transition_prob(z_process[index - 1], epsilon, delta) else 0
        distorted_encoding[index] = mod2_addition(int(bit), z_process[index])
        
      distorted_encoding[index] = str(distorted_encoding[index])

    distorted_pairs.append((sample, ''.join(distorted_encoding)))
      
  return distorted_pairs

def run_lloyds_with_normal_samples_and_polya_transmission(codebook_lengths, channel_error_probabilities, num_samples):
  mu = 0
  sigma = 1
  
  delta = 0.4
  
  normal_source_samples = np.random.normal(mu, sigma, num_samples)
  
  distortions = [0] * len(codebook_lengths)
  centroids = [0] * len(codebook_lengths)
  bins = [0] * len(codebook_lengths)
  
  distorted_pairs = [0] * len(codebook_lengths)
  
  sample_to_centroid_map = [0] * len(codebook_lengths)
  
  for channel_error_probability in channel_error_probabilities:
    for i in range(len(codebook_lengths)):
      [centroids[i], bins[i], distortions[i]] = general_lloyds_algorithm(normal_source_samples, num_samples, channel_error_probability, codebook_lengths[i])
      
      distorted_pairs[i] = send_through_polya_channel(channel_error_probability, delta, bins[i], centroids[i])

  return centroids, bins, distortions

def main():
  channel_error_probabilities = [0, 0.01, 0.1, 0.5]
  codebook_lengths = [1, 2, 4, 8]
  num_samples = 50
  run_lloyds_with_normal_samples_and_polya_transmission(codebook_lengths, channel_error_probabilities, num_samples)

if __name__ == "__main__":
  main()