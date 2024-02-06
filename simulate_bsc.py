import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm
import matplotlib.pyplot as plt

INVALID_CENTROID = -999999

def simulate_bsc(original_bins, centroids, epsilon):
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

  # Simulate the BSC by flipping each bit with probability epsilon
  distorted_pairs = []
  for sample, encoding in sample_encoding_pairs:  
    distorted_encoding = ''.join(['1' if (bit == '0' and np.random.rand() < epsilon) 
                                  else '0' if (bit == '1' and np.random.rand() < epsilon) 
                                  else bit for bit in encoding])
    distorted_pairs.append((sample, distorted_encoding))

  # Map distorted encodings back to bins, correctly placing individual samples
  sample_to_centroid_map = []
  for sample, distorted_encoding in distorted_pairs:
    bin_number = binary_strings.index(distorted_encoding) if distorted_encoding in binary_strings else None
    if bin_number is not None:
      # Associate the sample with the centroid of the bin it has been placed into
      sample_to_centroid_map.append((sample, centroids[bin_number]))
    else:
      # Record that the sample got mapped to an invalid centroid during transmission over channel
      sample_to_centroid_map.append((0, INVALID_CENTROID))

  return sample_to_centroid_map

def calc_distortion_between_centroids_and_transmitted_samples(sample_to_centroid_map, num_samples, sigma):
  distortion = 0
  
  for (sample, centroid) in sample_to_centroid_map:
    if (sample == 0) and (centroid == INVALID_CENTROID):
      # if sample got lost during transmission use varience of the source as distortion
      distortion += sigma
    else:
      distortion += (sample - centroid)**2
      
  return distortion / num_samples


def run_lloyds_with_normal_samples_and_BSC_transmission(codebook_lengths, channel_error_probabilities, num_samples):
  mu = 0
  sigma = 1
  epsilon = 0.01
  
  normal_source_samples = np.random.normal(mu, sigma, num_samples)
  
  distortions = [0] * len(codebook_lengths)
  centroids = [0] * len(codebook_lengths)
  bins = [0] * len(codebook_lengths)
  
  fig1, ax1 = plt.subplots()
  ax1.set_xlabel('Codebook Size (n)')
  ax1.set_ylabel('Distortion')
  ax1.set_title('Distortion for n-length Codebook before Transmission (Normal)')
  
  fig2, ax2 = plt.subplots()
  ax2.set_xlabel('Codebook Size (n)')
  ax2.set_ylabel('Distortion')
  ax2.set_title('Distortion for n-length Codebook after Transmission (Normal)')
  
  new_distortions = [0] * len(codebook_lengths)
  sample_to_centroid_map = [0] * len(codebook_lengths)
  for channel_error_probability in channel_error_probabilities:
    for i in range(len(codebook_lengths)):
      [centroids[i], bins[i], distortions[i]] = general_lloyds_algorithm(normal_source_samples, num_samples, channel_error_probability, epsilon, codebook_lengths[i])
      # print('OLD DISTORTION: ', 'codebook length=', codebook_lengths[i], 'old distortions=', distortions, 'channel error prob=', channel_error_probability)
      
      sample_to_centroid_map[i] = simulate_bsc(bins[i], centroids[i], channel_error_probability)
      new_distortions[i] = calc_distortion_between_centroids_and_transmitted_samples(sample_to_centroid_map[i], num_samples, sigma)
      # print('NEW DISTORTION: ', 'codebook length=', codebook_lengths[i], 'new distortions=', new_distortions, 'channel error prob=', channel_error_probability)
      
    ax1.plot(codebook_lengths, distortions)
    ax2.plot(codebook_lengths, new_distortions)
  plt.show()
  
  return centroids, bins, distortions

def main():
  channel_error_probabilities = [0, 0.01, 0.1, 0.5]
  codebook_lengths = [1, 2, 4, 8]
  num_samples = 1000
  run_lloyds_with_normal_samples_and_BSC_transmission(codebook_lengths, channel_error_probabilities, num_samples)

if __name__ == "__main__":
  main()
