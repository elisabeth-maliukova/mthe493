import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm, conditional_probability, code_rate
import matplotlib.pyplot as plt
from simulate_bsc import simulate_bsc
import cv2
from pathlib import Path
import math

INVALID_CENTROID = -999999

def reshape_list(flat_list, structure_list):
  reshaped_list = []
  index = 0

  for sublist in structure_list:
      sublist_length = len(sublist)
      reshaped_list.append(flat_list[index:index + sublist_length])
      index += sublist_length

  return reshaped_list

#trains the quantizer
def train_quantizer(training_images, codebook_length, channel_error_probabilities):
  training_samples = np.concatenate([image.flatten() for image in training_images])
  num_samples = len(training_samples)
    
  centroids, bins, _ = general_lloyds_algorithm(training_samples, num_samples, channel_error_probabilities, codebook_length)
  
  return centroids, bins

def test_quantizer(test_image, centroids, channel_error_probability, codebook_length):
  flattened_test_image = test_image.flatten()
  test_sample_to_centroid = []
  #finds the associated nearest centroid for each of the pixel values
  for sample in flattened_test_image:
    distortion = -1
    index = 0 # bin to place a sample into
    for i in range(0, codebook_length):
      new_distortion = 0
      for j in range(0, codebook_length):
        new_distortion += conditional_probability(i, j, channel_error_probability, code_rate(codebook_length)) * (sample - centroids[j])**2
      if new_distortion < distortion or distortion == -1:
        distortion = new_distortion
        index = i
    test_sample_to_centroid.append(centroids[index])      
  num_centroids = len(centroids)

  #the following chunk of code simulates the quantized image being passed through the BSC
  # Calculate the number of bits needed to represent the centroids
  num_bits = int(np.ceil(np.log2(num_centroids)))

  # Generate the binary strings for each centroid
  binary_strings = [format(i, f'0{num_bits}b') for i in range(num_centroids)]
  sample_encoding_pairs = []
  for centroid_val in test_sample_to_centroid:
    sample_encoding_pairs.append(binary_strings[centroids.index(centroid_val)])
  # Simulate the BSC by flipping each bit with probability epsilon
  distorted_binary_centroids = []
  for binary_cent in sample_encoding_pairs:  
    distorted_encoding = ''.join(['1' if (bit == '0' and np.random.rand() < channel_error_probability) 
                                  else '0' if (bit == '1' and np.random.rand() < channel_error_probability) 
                                  else bit for bit in binary_cent])
    distorted_binary_centroids.append(distorted_encoding)

  # Map distorted encodings into a list of decoded quantization levels, correctly placing individual samples
  centroid_decimal = []
  for distorted_encoding in distorted_binary_centroids:
    bin_number = binary_strings.index(distorted_encoding) if distorted_encoding in binary_strings else None
    if bin_number is not None:
      # Associate the sample with the centroid of the bin it has been placed into
      centroid_decimal.append(math.floor(centroids[bin_number]))
    else:
      # Record that the sample got mapped to an invalid centroid during transmission over channel
      centroid_decimal.append(INVALID_CENTROID)
      
    
  # Reshape the quantized test image to the original shape
  centroid_decimal = np.asarray(reshape_list(centroid_decimal,test_image), dtype=np.uint8)

  return centroid_decimal

def main():
  channel_error_probability = 0.5
  dir = "elephants"
  images = Path(dir).glob('*.jpg')
  training_images = []
  for image in images:
    training_images.append(cv2.resize(cv2.imread(str(dir+"/"+image.name), cv2.IMREAD_GRAYSCALE),(194, 289)))
  codebook_length = 4
  centroids, bins = train_quantizer(training_images, codebook_length, channel_error_probability)

  # Testing phase
  new_image = cv2.resize(cv2.imread("test/elephant_test.jpg", cv2.IMREAD_GRAYSCALE),(194, 289))
  # Display the original and quantized images side by side
  reconstructed_image = test_quantizer(new_image, centroids, channel_error_probability, codebook_length)

  # Display the original and reconstructed images
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(new_image, cmap='gray')
  plt.title('Original Image')

  plt.subplot(1, 2, 2)
  plt.imshow(reconstructed_image, cmap='gray')
  plt.title('Reconstructed Image')

  plt.show()

if __name__ == "__main__":
  main()
