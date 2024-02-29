import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import math
import random
import statistics
from scipy.fftpack import dct, idct

INVALID_CENTROID = -999999

# temporary table for testing purposes
#BIT_ALLOCATION_76BPP = [[4, 4, 4, 4, 4, 0, 0, 0]] * 8


BIT_ALLOCATION_76BPP = [[8, 7, 6, 4, 3, 0, 0, 0],
                        [7, 6, 5, 4, 0, 0, 0, 0],
                        [6, 5, 4, 0, 0, 0, 0, 0],
                        [4, 4, 0, 0, 0, 0, 0, 0],
                        [3, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]
                                                
BIT_ALLOCATION_58BPP = [[8, 7, 6, 4, 0, 0, 0, 0],
                        [7, 6, 5, 0, 0, 0, 0, 0],
                        [6, 5, 0, 0, 0, 0, 0, 0],
                        [4, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]

BIT_ALLOCATION_24BPP = [[8, 8, 0, 0, 0, 0, 0, 0],
                        [8, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]

# Shifts image pixel values to obtain a zero mean source
def translate_image(test_image, shift):
  mean_adjusted_image = [ [0] * len(test_image[0]) for _ in range(len(test_image))]
  for i in range(len(test_image)):
    for j in range(len(test_image[0])):
      mean_adjusted_image[i][j] = test_image[i][j] + shift
      if mean_adjusted_image[i][j] > 256:
        mean_adjusted_image[i][j] = 256 # If pixel translation is higher than 256, then bound it by 256
  return mean_adjusted_image

# Partions image into 8 by 8 squares to prepare for DCT 
def partition_image(translated_image):
  partitioned_image = []

  # move 8 by 8 square over image
  for x in range( (int) (len(translated_image[0]) / 8)):
    for y in range( (int) (len(translated_image) / 8) ):
      dct_square = [[0] * 8 for _ in range(8)]

      # save each pixel in 8 by 8 square
      for i in range(8):
        for j in range(8):
          dct_square[i][j] = translated_image[i + (x * 8)][j + y * 8]
      partitioned_image.append(dct_square)    
  
  return partitioned_image

# Rebuild image from partitioned blocks
def reconstruct_image(inverse_DCT_transform):
  reconstructed_image = []
  
  for y in range(32):
    for i in range(8):
      for x in range(32):
        for j in range(8):
          reconstructed_image.append(inverse_DCT_transform[i][j][x + 32 * y])

  reconstructed_image = np.array(reconstructed_image).reshape((256,256))
  return reconstructed_image

# performs DCT transformation on image
def DCT_transform_image(partitioned_image):
  num_blocks = len(partitioned_image)

  # Create 8x8x(num_blocks) array
  DCT_transform = np.array([[[0] * num_blocks] * 8 for _ in range(8)], dtype=np.float32)
  
  # Perform DCT transform on each block
  for k in range(num_blocks):
    dct_block = dct(dct(np.array(partitioned_image[k], dtype=np.float32).T, norm='ortho').T, norm='ortho')
    for i in range(8):
      for j in range(8):
        DCT_transform[i][j][k] = dct_block[i][j]

  return DCT_transform

# Perform inverse DCT transform on supplied matrix
def inverse_DCT_transform_image(DCT_transform):
  num_blocks = len(DCT_transform[0][0])
  inverse_DCT_transform = np.array([[[0] * num_blocks] * 8 for _ in range(8)], dtype=np.float64)

  for k in range(num_blocks):
    dct_block = DCT_transform[:, :, k]
    inverse_dct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
    for i in range(8):
      for j in range(8):
        inverse_DCT_transform[i][j][k] = inverse_dct_block[i][j] 
        
  return inverse_DCT_transform

# Determine variances of DCT Coefficients
def get_DCT_variances(DCT_transform):
  DCT_variances = np.array([[0] * 8 for _ in range(8)], dtype=np.float64)

  # Calculate Varience Coefficients
  for i in range(8):
    for j in range(8):
      DCT_variances[i][j] = statistics.variance(DCT_transform[i][j])
  return DCT_variances     

# Determine the ac quantizer rates
def get_AC_coefficent_rates():
  bit_allocation_map = BIT_ALLOCATION_76BPP

  flat_map = [item for sublist in bit_allocation_map for item in sublist]
  flat_map[0] = 0

  # Use a set to get unique elements
  AC_coefficient_rates = sorted(set(flat_map))

  return AC_coefficient_rates

# Determine DC quantizer rates
def get_DC_coefficent_rates():
  bit_allocation_map = BIT_ALLOCATION_76BPP
  return [bit_allocation_map[0][0]]

# train quantizers using optmial bit allocation rates
def train_quantizers(source, code_rates, channel_error_probability, num_samples):
  quantizers = [[0], [0], [0], [0], [0], [0], [0], [0], [0]]

  print("Starting quantizer training")
  for i in range(1, 9):
    if i in code_rates:
      print("starting rate", i)
      codebook_length = 2**i
      [quantizers[i], _, _ ] = general_lloyds_algorithm(source, num_samples, channel_error_probability, codebook_length)
  print("Finished quantizer training")      
  return quantizers

  
def encode_coefficients(standard_quantizers, coefficients, rate):
  coefficients_len = len(coefficients)
  encoded_coefficients = [None] * coefficients_len

  if rate == 0:
    encoded_coefficients = [0] * coefficients_len
  else:    
    codebook_length = 2**rate


    for i in range(coefficients_len):
      distortion = math.inf
      for j in range(codebook_length):
        if (standard_quantizers[rate][j] - coefficients[i])**2 < distortion:
          distortion = (standard_quantizers[rate][j] - coefficients[i])**2
          encoded_coefficients[i] = j

  return encoded_coefficients

def decode_coefficients(standard_quantizers, encodeings, rate):
  coefficients_len = len(encodeings)
  decoded_coefficients = np.array([0] * coefficients_len, dtype=np.float64)

  
  for i in range(coefficients_len):
    if rate != 0: 
      decoded_coefficients[i] = standard_quantizers[rate][encodeings[i]]
    else:
      decoded_coefficients[i] = 0

  return decoded_coefficients

# Quantize DCT trasnform
def encode_DCT_transform(standard_normal_quantizers, standard_laplace_quantizers, DCT_transform, DCT_variences):
  num_blocks = len(DCT_transform[0][0])
  varience_adjusted_values = np.array([[[0] * num_blocks] * 8 for _ in range(8)], dtype=np.float64)
  encoded_values = np.array([[[0] * num_blocks] * 8 for _ in range(8)])

  # adjust values for variences
  for i in range(8):
    for j in range(8):
      for k in range(num_blocks):
          varience_adjusted_values[i][j][k] = np.float64(DCT_transform[i][j][k] / math.sqrt(DCT_variences[i][j]))

  for i in range(8):
    for j in range(8):
      if i == 0 and j == 0:
        encoded_values[i][j] = encode_coefficients(standard_normal_quantizers, varience_adjusted_values[0][0], BIT_ALLOCATION_76BPP[0][0])
      else:
        encoded_values[i][j] = encode_coefficients(standard_laplace_quantizers, varience_adjusted_values[i][j], BIT_ALLOCATION_76BPP[i][j])
  return encoded_values

def decode_DCT_transform(transmitted_values, standard_normal_quantizers, standard_laplace_quantizers, DCT_variences):
  num_blocks = len(transmitted_values[0][0])
  decoded_values = np.array([[[0] * num_blocks] * 8 for _ in range(8)], dtype=np.float64) 

  for i in range(8):
    for j in range(8):
      if i == 0 and j == 0:
        decoded_values[i][j] = decode_coefficients(standard_normal_quantizers, transmitted_values[0][0], BIT_ALLOCATION_76BPP[0][0])
      else:
        decoded_values[i][j] = decode_coefficients(standard_laplace_quantizers, transmitted_values[i][j], BIT_ALLOCATION_76BPP[i][j])
      
  for i in range(8):
    for j in range(8):
      for k in range(num_blocks):
        decoded_values[i][j][k] = decoded_values[i][j][k] * math.sqrt(DCT_variences[i][j])

  return decoded_values 


def flip_bits_with_probability(number, epsilon, rate):
    # Convert the integer to binary representation
    binary_representation = bin(number)[2:].zfill(rate)

    # Flip each bit with probability epsilon
    flipped_binary = ''.join(
        str(int(bit) ^ (random.random() < epsilon))
        for bit in binary_representation
    )

    # Convert the flipped binary back to an integer
    flipped_number = int(flipped_binary, 2)

    return flipped_number      

def simulate_channel(encoded_values, error_probability):
  num_blocks = len(encoded_values[0][0])
  sent_values = np.array([[[0] * num_blocks] * 8 for _ in range(8)]) 
  for i in range(8):
    for j in range(8):
      rate = BIT_ALLOCATION_76BPP[i][j]
      if rate != 0:
        for k in range(len(encoded_values[i][j])):
          sent_values[i][j][k] = flip_bits_with_probability(encoded_values[i][j][k], error_probability, rate)
  
  return sent_values      

def create_quantizers(channel_error_probabilities):
  num_samples = 10**3
  mean = 0
  variance = 1
  
  # Create Zero mean, Unit variance laplace and normal sources.
  normal_source_samples = np.random.normal(0, variance, num_samples)
  laplacian_source_samples = np.random.laplace(mean, variance / math.sqrt(2), num_samples)

  DC_coefficent_rate = get_DC_coefficent_rates()
  AC_coefficent_rates = get_AC_coefficent_rates()
  
  # Train Quantizers for laplace and normal sources
  standard_normal_quantizers = train_quantizers(normal_source_samples, DC_coefficent_rate, channel_error_probabilities, num_samples)
  standard_laplace_quantizers = train_quantizers(laplacian_source_samples, AC_coefficent_rates, channel_error_probabilities, num_samples)

  return [standard_normal_quantizers, standard_laplace_quantizers]  

def main():
  dir = "elephants"
  images = Path(dir).glob('*.jpg')
  training_images = []
  for image in images:
    training_images.append(cv2.resize(cv2.imread(str(dir+"/"+image.name), cv2.IMREAD_GRAYSCALE),(256, 256)))

  channel_error_probabilities = 0.01
  [standard_normal_quantizer, standard_laplace_quantizers] = create_quantizers(channel_error_probabilities) 

  translated_image = translate_image(training_images[1], -128)
  partitioned_image = partition_image(translated_image)
  DCT_transform = DCT_transform_image(partitioned_image)
  DCT_variances = get_DCT_variances(DCT_transform)
  
  encoded_DCT_transform = encode_DCT_transform(standard_normal_quantizer, standard_laplace_quantizers, DCT_transform, DCT_variances)
  transmitted_DCT_transfrom = simulate_channel(encoded_DCT_transform, channel_error_probabilities)
  decoded_DCT_transform = decode_DCT_transform(transmitted_DCT_transfrom, standard_normal_quantizer, standard_laplace_quantizers, DCT_variances)


  inverse_DCT_transform = inverse_DCT_transform_image(decoded_DCT_transform)
  reconstructed_image = reconstruct_image(inverse_DCT_transform)
  final_image = translate_image(reconstructed_image, 128)

  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(training_images[1], cmap='gray')
  plt.title('Original Image')

  plt.subplot(1, 2, 2)
  plt.imshow(final_image, cmap='gray')
  plt.title('Reconstructed Image')

  plt.show()

if __name__ == "__main__":
  main()
