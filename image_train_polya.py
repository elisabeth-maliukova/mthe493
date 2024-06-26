import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm, conditional_probability_polya, code_rate
from polya_channel import simulate_polya_channel, get_transition_prob, mod2_addition 
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
def train_quantizer(training_images, codebook_length, channel_error_probabilities,delta):
  training_samples = np.concatenate([image.flatten() for image in training_images])
  num_samples = len(training_samples)
    
  centroids, bins, _ = general_lloyds_algorithm(training_samples, num_samples, channel_error_probabilities, codebook_length, 'polya', delta)

  return centroids, bins

def test_quantizer(test_image, centroids, channel_error_probability, codebook_length, delta):
  flattened_test_image = test_image.flatten()
  test_sample_to_centroid = []
  #finds the associated nearest centroid for each of the pixel values
  for sample in flattened_test_image:
    distortion = -1
    index = 0 # bin to place a sample into
    for i in range(0, codebook_length):
      new_distortion = 0
      for j in range(0, codebook_length):
        iBits = "{0:b}".format(i).zfill(code_rate(codebook_length))
        jBits = "{0:b}".format(j).zfill(code_rate(codebook_length))
        new_distortion += conditional_probability_polya(iBits, jBits, channel_error_probability, delta) * (sample - centroids[j])**2
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
  distorted_pairs = []

  for encoding in sample_encoding_pairs:
    distorted_encoding = [0] * len(encoding)
    z_process = [0] * len(encoding)
    z_process[0] = 1 if (np.random.rand() < channel_error_probability) else 0
    
    for index, bit in enumerate(encoding):
      if (index == 0):
        distorted_encoding[0] = mod2_addition(int(bit), z_process[0])
      else:
        z_process[index] = 1 if np.random.rand() < get_transition_prob(z_process[index - 1], channel_error_probability, delta) else 0
        distorted_encoding[index] = mod2_addition(int(bit), z_process[index])
        
      distorted_encoding[index] = str(distorted_encoding[index])

    distorted_pairs.append(''.join(distorted_encoding))
      
  #distorted_pairs = simulate_polya_channel(channel_error_probability, delta, sample_encoding_pairs)


  # Map distorted encodings into a list of decoded quantization levels, correctly placing individual samples
  centroid_decimal = []
  for distorted_encoding in distorted_pairs:
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
  delta = 5
  channel_error_probability = 0.1
  dir = "elephants"
  images = Path(dir).glob('*.jpg')
  training_images = []
  for image in images:
    training_images.append(cv2.resize(cv2.imread(str(dir+"/"+image.name), cv2.IMREAD_GRAYSCALE),(194, 289)))
  codebook_length = 4
  centroids, bins = train_quantizer(training_images, codebook_length, channel_error_probability,delta)

  # Testing phase
  new_image = cv2.resize(cv2.imread("test/elephant_test.jpg", cv2.IMREAD_GRAYSCALE),(194, 289))
  # Display the original and quantized images side by side
  reconstructed_image = test_quantizer(new_image, centroids, channel_error_probability, codebook_length, delta)

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
