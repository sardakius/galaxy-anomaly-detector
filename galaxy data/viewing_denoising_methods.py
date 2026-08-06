import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

DATASETS = ["Raw", "Visu Wavelet", "Bayes Wavelet", "Gaussian 0.5σ", "Gaussian 1σ", "Gaussian 2σ", "Noise2Void"]
images = []

for dataset in DATASETS:
  path = os.path.join(f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/{dataset.lower().replace('.', '_')}/normal galaxies", 'galaxy_59_89994238064714_-49_03071064588237.png')
  img = Image.open(path).convert('L')
  img = img.resize((80, 80))
  img = np.expand_dims(img, axis=-1)
  images.append(np.array(img, dtype=np.float32))

# Set figure size (12x6 is a clean aspect ratio for square grids)
fig = plt.figure(figsize=(12, 6))

# --- Precise Geometric Coordinates (0.0 to 1.0 of the canvas) ---
img_w = 0.18        # Width of each image
img_h = 0.36        # Height of each image (keeps them square matching the 12:6 fig ratio)
gap_x = 0.02        # Tight horizontal gap between images
gap_y = 0.07        # Reduced vertical gap to bring rows closer together

# Calculate total row widths to center them horizontally
width_top_row = (3 * img_w) + (2 * gap_x)
width_bottom_row = (4 * img_w) + (3 * gap_x)

start_x_top = (1.0 - width_top_row) / 2
start_x_bottom = (1.0 - width_bottom_row) / 2

# Row Y positions (brought much closer around the vertical center 0.5)
start_y_top = 0.5 + (gap_y / 2)
start_y_bottom = 0.5 - (gap_y / 2) - img_h

# --- Render Top Row (3 Images) ---
for i in range(3):
    x = start_x_top + i * (img_w + gap_x)
    ax = fig.add_axes([x, start_y_top, img_w, img_h])
    ax.imshow(images[i]) # Correct grayscale mapping
    ax.set_title(DATASETS[i])
    ax.axis('off')

# --- Render Bottom Row (4 Images) ---
for i in range(4):
    x = start_x_bottom + i * (img_w + gap_x)
    ax = fig.add_axes([x, start_y_bottom, img_w, img_h])
    ax.imshow(images[i + 3]) # Correct grayscale mapping
    ax.set_title(DATASETS[i + 3])
    ax.axis('off')

plt.suptitle("Different Denoising Methods")
plt.show()
