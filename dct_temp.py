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
def translate_image(test_image):
  mean_adjusted_image = [ [0] * len(test_image[0]) for _ in range(len(test_image))]
  for i in range(len(test_image)):
    for j in range(len(test_image[0])):
      mean_adjusted_image[i][j] = test_image[i][j] - 128
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

# performs DCT transformation on image
def DCT_transform_image(partitioned_image):
  num_blocks = len(partitioned_image)
  DCT_transform = np.array([[[0] * num_blocks] * 8 for _ in range(8)], dtype=np.float32)
  
  for k in range(num_blocks):
    dct_block = np.array(partitioned_image[k], dtype=np.float32) 
    for i in range(8):
      for j in range(8):
        DCT_transform[i][j][k] = dct_block[i][j]

  return DCT_transform

def main():
  channel_error_probability = 0.5
  dir = "elephants"
  images = Path(dir).glob('*.jpg')
  training_images = []
  for image in images:
    training_images.append(cv2.resize(cv2.imread(str(dir+"/"+image.name), cv2.IMREAD_GRAYSCALE),(256, 256)))

  translated_image = translate_image(training_images[0])
  partitioned_image = partition_image(translated_image)
  DCT_transform = DCT_transform_image(partitioned_image)

if __name__ == "__main__":
  main()
