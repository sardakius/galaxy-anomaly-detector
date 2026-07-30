from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from astropy.visualization import ImageNormalize, PercentileInterval, LinearStretch

from PIL import Image
import os

folder = "/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/raw/normal galaxies"
target = "/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/gaussian 2σ/normal galaxies"

for filename in os.listdir(folder):
    if filename.lower().endswith(".png") and not os.path.isfile(os.path.join(target, filename)):
        filepath = os.path.join(folder, filename)
        
        img = Image.open(filepath)
        img = img.convert("L")

        img_arr = np.array(img)

        denoised_img_arr = ndimage.gaussian_filter(img_arr, sigma=2)
        denoised_img = Image.fromarray(denoised_img_arr)

        denoised_img.save(os.path.join(target, filename))

        print(f"Processed {filename}")

print("Finished denoising images.")