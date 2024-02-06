import numpy as np
from general_lloyd_algorithm import general_lloyds_algorithm
import matplotlib.pyplot as plt
from simulate_bsc import simulate_bsc
import cv2
from pathlib import Path


INVALID_CENTROID = -999999

#trains the quantizer
def train_quantizer(training_images, codebook_length):
  mu = 0
  sigma = 1
  epsilon = 0.01
  print(training_images)
  training_samples = np.concatenate([image.flatten() for image in training_images])
  num_samples = len(training_samples)
    
  centroids, bins, _ = general_lloyds_algorithm(training_samples, num_samples, 0, epsilon, codebook_length)
    
  return centroids, bins

def test_quantizer(new_image, centroids, bins, channel_error_probability, epsilon):
  #num_samples = len(new_image.flatten())

  sample_encoding_pairs = simulate_bsc(bins, centroids, channel_error_probability)

  reconstructed_image = np.zeros_like(new_image, dtype=np.uint8)

  for (sample, centroid) in sample_encoding_pairs:
      if centroid != INVALID_CENTROID:
          reconstructed_image[np.isin(new_image, sample)] = centroid

  return reconstructed_image





def main():
  channel_error_probability = 0.01
  epsilon = 0.01
  # Training phase
  dir = "elephants"
  images = Path(dir).glob('*.jpg')
  training_images = []
  for image in images:
    training_images.append(cv2.resize(cv2.imread(str(dir+"/"+image.name), cv2.IMREAD_GRAYSCALE),(194, 289)))
  codebook_length = 4
  centroids, bins = train_quantizer(training_images, codebook_length)

  # Testing phase
  new_image = cv2.resize(cv2.imread("test/elephant_test.jpg", cv2.IMREAD_GRAYSCALE),(194, 289))
  # Display the original and quantized images side by side
  reconstructed_image = test_quantizer(new_image, centroids, bins, channel_error_probability, epsilon)

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
