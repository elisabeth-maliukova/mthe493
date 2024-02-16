import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm, conditional_probability, code_rate
import matplotlib.pyplot as plt
from simulate_bsc import simulate_bsc
import cv2
from pathlib import Path
import math
from scipy.fftpack import dct, idct
INVALID_CENTROID = -999999

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


## We need to figure out hwo to do this properly
def reconstruct_image(inverse_DCT_transform):
  reconstructed_image = []
  for y in range(32):
    for x in range(32):
      temp = []
      for i in range(8):
        temp.append(inverse_DCT_transform[y][i][x])
    reconstructed_image.append(temp)

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

def inverse_DCT_transform_image(DCT_transform):
  num_blocks = len(DCT_transform[0][0])
  inverse_DCT_transform = np.array([[[0] * num_blocks] * 8 for _ in range(8)], dtype=np.float32)

  for k in range(num_blocks):
    dct_block = DCT_transform[:, :, k]
    inverse_dct_block = idct(idct(dct_block, axis=0, norm='ortho'), axis=1, norm='ortho')
    for i in range(8):
      for j in range(8):
        inverse_DCT_transform[i][j][k] = inverse_dct_block[i][j] 
        
  return inverse_DCT_transform

# Determine variances of DCT Coefficients
def get_DCT_variances(DCT_transform):
  DCT_variances = [[0] * 8 for _ in range(8)]
  
  # Calculate Varience of DC Coefficients
  DCT_variances[0][0] = np.var(DCT_transform[0][0])

  # Calculate Varience (Scale paramater for Laplace Source) of AC Coefficients
  for i in range(1, 8):
    for j in range(1, 8):
      scale_param = np.median(np.abs(DCT_transform[i][j] - np.median(DCT_transform[i][j]))) / 0.6745
      DCT_variances[i][j] = scale_param 

  return DCT_variances     

def train_quantizer(source, codebook_lengths, channel_error_probabilities, num_samples):
  
  centroids = [0] * len(codebook_lengths)  
  for channel_error_probability in channel_error_probabilities:
    for i in range(len(codebook_lengths)):
      [centroids[i], _, _ ] = general_lloyds_algorithm(source, num_samples, channel_error_probability, codebook_lengths[i])
      
  return centroids

def main():
  channel_error_probabilities = [0.01];
  codebook_lengths = [1, 2, 4, 8]
  dir = "elephants"
  images = Path(dir).glob('*.jpg')
  training_images = []
  for image in images:
    training_images.append(cv2.resize(cv2.imread(str(dir+"/"+image.name), cv2.IMREAD_GRAYSCALE),(256, 256)))

  translated_image = translate_image(training_images[1], -128)
  partitioned_image = partition_image(translated_image)
  DCT_transform = DCT_transform_image(partitioned_image)
  DCT_variances = get_DCT_variances(DCT_transform)
  
  inverse_DCT_transform = inverse_DCT_transform_image(DCT_transform)
  
  reconstructed_image = reconstruct_image(inverse_DCT_transform)
  
  final_image = translate_image(reconstructed_image, 128)
  
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 2)
  plt.imshow(final_image, cmap='gray')
  plt.title('Reconstructed Image')  
  plt.show()
  
  
  # reconstructed_image - translate_image()
  
  # print(DCT_variances)
  # num_samples = 10**3
  
  # mu = 0
  # sigma = 1
  
  # normal_source_samples = np.random.normal(mu, sigma, num_samples)
  # standard_normal_quantizer = train_quantizer(normal_source_samples, codebook_lengths, channel_error_probabilities, num_samples)
  
  # mean = 0
  # scale = 1 / math.sqrt(2)
  
  # laplacian_source_samples = np.random.laplace(mean, scale, num_samples)
  # laplacian_quantizer = train_quantizer(laplacian_source_samples, codebook_lengths, channel_error_probabilities, num_samples)
  
  
  
  
  # Next Steps 
  #            --Determine Bit Allocation for each block using Tables in Julians thesis
  #            --Quantize each block using standard quantizers (adjust using DCT variances)
  #            --Send over Channel
  #            --Decode Quantizer (adjust using DCT variances)
  #            --Perform inverse DCT transformation
  #            --add 128 to each pixel value to reverse translation to zero mean source
  #            --rebuild image from blocks

if __name__ == "__main__":
  main()
