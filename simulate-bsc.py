import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm, calc_distortion_for_all_bins
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
      

def run_lloyds_with_normal_samples(codebook_lengths, channel_error_probability, num_samples):
  mu = 0
  sigma = 1
  epsilon = 0.01
  
  normal_source_samples = np.random.normal(mu, sigma, num_samples)
  channel_error_probability = 0.1
  [centroids, bins, distortion] = general_lloyds_algorithm(normal_source_samples, num_samples, channel_error_probability, epsilon, codebook_lengths)
  # codebook_length = []
  # distortion = []
  # centroids = []
  # bins = []
  # for i in range(rate):
  #   codebook_length.append(2**i)
  #   distortion.append(0)
  #   centroids.append(0)
  #   bins.append(0)
  # plt.figure()
  # plt.xlabel('Codebook Size (n)')
  # plt.ylabel('Distortion')
  # plt.title('Distortion for n-length Codebook (Normal)')
  # for error in channel_error_probability:
  #   for j in range(rate):
  #     [centroids[j], bins[j], distortion[j]] = general_lloyds_algorithm(normal_source_samples, num_samples, error, epsilon, codebook_length[j])
    # plt.plot(codebook_length, distortion)
  # plt.legend([str(channel_error_probability[0]),str(channel_error_probability[1]),str(channel_error_probability[2]),str(channel_error_probability[3])])
  return [normal_source_samples, centroids, bins, distortion]

def plot_distortions(codebook_lengths, new_distortion):
  plt.figure()
  plt.xlabel('Codebook Size (n)')
  plt.ylabel('Distortion')
  plt.title('Distortion for n-length Codebook after Transmission')
  plt.plot(codebook_lengths, new_distortion)
  plt.show()

def main():
  channel_error_probability = [0, 0.01, 0.1, 0.5]
  codebook_lengths = 2**2
  num_samples = 100
  normal_source_samples, centroids, original_bins, distortion = run_lloyds_with_normal_samples(codebook_lengths, channel_error_probability, num_samples)

  epsilon = 0.1  # Example flipping probability
  new_bins = simulate_bsc(original_bins, len(centroids), epsilon)
  
  new_distortion = calc_distortion_for_all_bins(new_bins, centroids, codebook_lengths, num_samples, epsilon)
  
  plot_distortions(codebook_lengths, new_distortion)


if __name__ == "__main__":
  main()
