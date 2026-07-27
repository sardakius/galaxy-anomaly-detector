from PIL import Image
import os

folder = "."

for filename in os.listdir(folder):
    if filename.lower().endswith(".png"):

        filepath = os.path.join(folder, filename)

        # Open image
        img = Image.open(filepath)

        # Convert to grayscale
        img = img.convert("L")

        # Resize to 80x80
        img = img.resize((80, 80), Image.Resampling.LANCZOS)

        # Overwrite original
        img.save(filepath)

        print(f"Processed {filename}")

print("Finished converting and resizing all images.")