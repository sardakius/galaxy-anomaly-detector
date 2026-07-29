import os
import numpy as np
from PIL import Image
from astropy.visualization import LinearStretch, ImageNormalize, PercentileInterval

image_paths = []
all_pixels = []

for folder in [
    "normal galaxies",
    "anomalous galaxies",
    "unknown galaxies"
]:
    for filename in os.listdir(folder):
        if filename.lower().endswith(".png"):
                path = os.path.join(folder, filename)
                img = np.array(Image.open(path), dtype=np.float32)

                image_paths.append(path)
                all_pixels.append(img.ravel())

# Compute one normalization for the entire dataset
all_pixels = np.concatenate(all_pixels)

norm = ImageNormalize(
    all_pixels,
    interval=PercentileInterval(98),
    stretch=LinearStretch()
)


# Normalize and overwrite every image
for path in image_paths:
    img = np.array(Image.open(path), dtype=np.float32)

    normalized = norm(img) 
    normalized = (normalized * 255).astype(np.uint8)

    Image.fromarray(normalized).save(path)