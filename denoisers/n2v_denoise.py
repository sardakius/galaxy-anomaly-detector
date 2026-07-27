import os
import numpy as np
from skimage.io import imread, imsave
from n2v.models import N2V, N2VConfig

# =====================================================
# Directories
# =====================================================

train_dir = "/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/raw/normal galaxies"
output_dir = "/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/noise2void/normal galaxies"
model_dir = "models"
model_name = "euclid_n2v"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# =====================================================
# Load training images
# =====================================================

images = []
filenames = []

for filename in sorted(os.listdir(train_dir)):
    if filename.lower().endswith(".png"):

        img = imread(os.path.join(train_dir, filename))

        # If RGB, convert to grayscale
        if img.ndim == 3:
            img = img[..., 0]

        img = img.astype(np.float32)

        # Normalize to [0,1]
        img -= img.min()
        img /= (img.max() + 1e-8)

        images.append(img)
        filenames.append(filename)

images = np.stack(images)
images = images[..., np.newaxis]

print(f"Loaded {len(images)} images.")
print(images.shape)

# =====================================================
# Configure N2V
# =====================================================

config = N2VConfig(
    X=images,
    train_steps_per_epoch=200,
    train_epochs=100,
    train_batch_size=32,
    train_loss="mse",
    batch_norm=True,
    unet_kern_size=3,
    n2v_patch_shape=(48, 48),   # Better for 80x80 images
    n2v_perc_pix=0.198,
    n2v_manipulator="uniform_withCP",
    n2v_neighborhood_radius=5,
)

# =====================================================
# Train model
# =====================================================

model = N2V(
    config=config,
    name=model_name,
    basedir=model_dir,
)

model.train(images, images)

print("Training complete.")

# =====================================================
# Denoise every image
# =====================================================

for filename in filenames:
    if filename.lower().endswith(".png"):
        print(f"Processing {filename}")
        img = imread(os.path.join(train_dir, filename))

        if img.ndim == 3:
            img = img[..., 0]

        img = img.astype(np.float32)

        img -= img.min()
        img /= (img.max() + 1e-8)

        prediction = model.predict(
            img[..., np.newaxis],
            axes="YXC"
        )

        prediction = np.squeeze(prediction)

        # Save back as 8-bit PNG
        prediction = (prediction * 255).clip(0, 255).astype(np.uint8)

        imsave(
            os.path.join(output_dir, filename),
            prediction
        )

print("Done!")