from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from astropy.visualization import ImageNormalize, PercentileInterval, LinearStretch

img = Image.open('normal galaxies/galaxy63788115839234955-45912051109118394.png').convert('L').resize((80, 80))
img_array = np.array(img)

print(img.size)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

axes[0].imshow(img_array, cmap='gray')
axes[0].set_title("Original")
axes[0].axis("off")

from PIL import Image
import os

folder = "normal galaxies"
half_sigma_folder = "gaussian0_5/normal galaxies"
full_sigma_folder = "gaussian1/normal galaxies"

for filename in os.listdir(folder):
    if filename.lower().endswith(".png") and not os.path.isfile(os.path.join(half_sigma_folder, filename)) and not os.path.isfile(os.path.join(full_sigma_folder, filename)):
        filepath = os.path.join(folder, filename)
        
        img = Image.open(filepath)
        img = img.convert("L")

        img_arr = np.array(img)

        denoised_05_img_arr = ndimage.gaussian_filter(img_arr, sigma=0.5)
        denoised_05_img = Image.fromarray(denoised_05_img_arr)

        denoised_1_img_arr = ndimage.gaussian_filter(img_arr, sigma=1.0)
        denoised_1_img = Image.fromarray(denoised_1_img_arr)

        denoised_05_img.save(os.path.join(half_sigma_folder, filename))
        denoised_1_img.save(os.path.join(full_sigma_folder, filename))

        print(f"Processed {filename}")

print("Finished denoising images.")