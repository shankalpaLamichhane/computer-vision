# Approach 3 

import numpy as np
from PIL import Image

# Define constants
learning_rate = 0.1
num_iterations = 100
codebook_dim = 16
dither_pattern_size = 3
dither_pattern = np.array([[0, 7, 3], [6, 5, 2], [4, 1, 8]])

# Load image and convert to grayscale
img = Image.open("LENNA.bmp")
img = img.convert("L")
width, height = img.size

# Convert image to numpy array
img_array = np.array(img)

# Initialize codebook
codebook = np.random.rand(codebook_dim**2, dither_pattern_size, dither_pattern_size)

# Train SOM
for iteration in range(num_iterations):
    for i in range(height // dither_pattern_size):
        for j in range(width // dither_pattern_size):
            # Compute indices of dither pattern
            row = np.mod(np.arange(i*dither_pattern_size, (i+1)*dither_pattern_size), dither_pattern_size)
            col = np.mod(np.arange(j*dither_pattern_size, (j+1)*dither_pattern_size), dither_pattern_size)

            # Extract dither pattern from image
            pattern = img_array[i*dither_pattern_size : (i+1)*dither_pattern_size, j*dither_pattern_size : (j+1)*dither_pattern_size]

            # Find best matching codebook vector
            distances = np.sum((pattern - codebook)**2, axis=(1,2))
            idx = np.argmin(distances)

            # Update codebook vector
            codebook[idx] += learning_rate * (pattern - codebook[idx])

# Quantize image using codebook
quantized_img = np.zeros_like(img_array)
for i in range(height // dither_pattern_size):
    for j in range(width // dither_pattern_size):
        # Compute indices of dither pattern
        row = np.mod(np.arange(i*dither_pattern_size, (i+1)*dither_pattern_size), dither_pattern_size)
        col = np.mod(np.arange(j*dither_pattern_size, (j+1)*dither_pattern_size), dither_pattern_size)

        # Extract dither pattern from image
        pattern = img_array[i*dither_pattern_size : (i+1)*dither_pattern_size, j*dither_pattern_size : (j+1)*dither_pattern_size]

        # Find best matching codebook vector
        distances = np.sum((pattern - codebook)**2, axis=(1,2))
        idx = np.argmin(distances)

        # Quantize pixel values using codebook vector
        quantized_pattern = codebook[idx]
        quantized_img[i*dither_pattern_size : (i+1)*dither_pattern_size, j*dither_pattern_size : (j+1)*dither_pattern_size] = quantized_pattern[row, col]

# Create compressed image
compressed_img = Image.fromarray(quantized_img.astype("uint8"))

# Save compressed image
compressed_img.save("example_compressed.png")

import matplotlib.pyplot as plt
import cv2

img = cv2.imread('./LENNA.bmp')

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original Image")
axes[1].imshow(compressed_img, cmap="gray")
axes[1].set_title("Compressed Image")
plt.show()