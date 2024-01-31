import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm
import matplotlib.pyplot as plt

def simulate_bsc(original_bins, num_centroids, epsilon):
  # Calculate the number of bits needed to represent the centroids
  num_bits = int(np.ceil(np.log2(num_centroids)))
  
  # Generate the binary strings for each centroid
  binary_strings = [format(i, f'0{num_bits}b') for i in range(num_centroids)]
  
  # Associate each sample with its encoding directly
  sample_encoding_pairs = []
  for bin_index, bin in enumerate(original_bins):
    for sample in bin:
      sample_encoding_pairs.append((sample, binary_strings[bin_index]))

  # Simulate the BSC by flipping each bit with probability epsilon
  distorted_pairs = []
  for sample, encoding in sample_encoding_pairs:
    distorted_encoding = ''.join(['1' if (bit == '0' and np.random.rand() < epsilon) 
                                  else '0' if (bit == '1' and np.random.rand() < epsilon) 
                                  else bit for bit in encoding])
    distorted_pairs.append((sample, distorted_encoding))

  # Initialize new bins for the distorted samples
  new_bins = [[] for _ in range(num_centroids)]
  
  # Map distorted encodings back to bins, this time correctly placing individual samples
  for sample, distorted_encoding in distorted_pairs:
    bin_number = binary_strings.index(distorted_encoding) if distorted_encoding in binary_strings else None
    if bin_number is not None:
      new_bins[bin_number].append(sample)

  return new_bins

def calc_distortion_between_bin_and_transmitted_sample(new_bins, centroids, codebook_length, num_samples):
  distortion = 0
  
  for i in range(codebook_length):
    for sample in new_bins[i]:
      distortion += (sample - centroids[i])**2
      
  return distortion / num_samples


def run_lloyds_with_normal_samples_and_BSC_transmission(codebook_lengths, channel_error_probabilities, num_samples):
  mu = 0
  sigma = 1
  epsilon = 0.01
  
  normal_source_samples = np.random.normal(mu, sigma, num_samples)
  
  distortions = [0] * len(codebook_lengths)
  centroids = [0] * len(codebook_lengths)
  bins = [0] * len(codebook_lengths)
  
  plt.figure()
  plt.xlabel('Codebook Size (n)')
  plt.ylabel('Distortion')
  plt.title('Distortion for n-length Codebook (Normal)')
  
  new_bins = [0] * len(codebook_lengths)
  new_distortions = [0] * len(codebook_lengths)
  for channel_error_probability in channel_error_probabilities:
    for i in range(len(codebook_lengths)):
      [centroids[i], bins[i], distortions[i]] = general_lloyds_algorithm(normal_source_samples, num_samples, channel_error_probability, epsilon, codebook_lengths[i])
      print('OLD DISTORTION: ', 'codebook length=', codebook_lengths[i], 'old distortions=', distortions, 'channel error prob=', channel_error_probability)
      
      new_bins[i] = simulate_bsc(bins[i], len(centroids[i]), channel_error_probability)
      new_distortions[i] = calc_distortion_between_bin_and_transmitted_sample(new_bins[i], centroids[i], codebook_lengths[i], num_samples)
      print('NEW DISTORTION: ', 'codebook length=', codebook_lengths[i], 'new distortions=', new_distortions, 'channel error prob=', channel_error_probability)
      
    # *****UNCOMMENT BELOW LINE TO ADD THE GRAPHS OF THE DISTORTIONS BEFORE TRANSMISSION*****
    # plt.plot(codebook_lengths, distortions)
    
    plt.plot(codebook_lengths, new_distortions)
    
  plt.legend([str(channel_error_probabilities[0]),str(channel_error_probabilities[1]),str(channel_error_probabilities[2]),str(channel_error_probabilities[3])])
  
  plt.show()
  
  return centroids, bins, distortions

def main():
  channel_error_probabilities = [0, 0.01, 0.1, 0.5]
  codebook_lengths = [1, 2, 4, 8]
  num_samples = 1000
  run_lloyds_with_normal_samples_and_BSC_transmission(codebook_lengths, channel_error_probabilities, num_samples)

if __name__ == "__main__":
  main()
