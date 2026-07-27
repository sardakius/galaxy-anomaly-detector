from skimage.restoration import denoise_wavelet
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

folder = "raw/anomalous galaxies"
wavelet_folder = "visu wavelet/anomalous galaxies"

for filename in os.listdir(folder):
    if filename.lower().endswith(".png") and not os.path.isfile(os.path.join(wavelet_folder, filename)):
        filepath = os.path.join(folder, filename)
        
        img = Image.open(filepath)
        img_arr = np.array(img)

        im_denoised = denoise_wavelet(
            img_arr,
            method='VisuShrink',
            mode='soft',
            wavelet_levels=3,
            rescale_sigma=True,
            channel_axis=None
        )

        im_denoised = (im_denoised * 255).astype(np.uint8)  # Convert back to uint8

        denoised_img = Image.fromarray(im_denoised)
        denoised_img.save(os.path.join(wavelet_folder, filename))

        print(f"Processed {filename}")

print("Finished denoising images.")
