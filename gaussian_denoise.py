from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from astropy.visualization import ImageNormalize, PercentileInterval, LinearStretch

from PIL import Image
import os

folder = "raw/anomalous galaxies"
half_sigma_folder = "gaussian0_5/anomalous galaxies"
full_sigma_folder = "gaussian1/anomalous galaxies"

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