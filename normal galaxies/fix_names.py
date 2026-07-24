from PIL import Image
import os

folder = "."

for filename in os.listdir(folder):
    print(filename, filename.lower().find("y_") == -1 and filename.lower().endswith(".png"))
    if filename.lower().find("y_") == -1 and filename.lower().endswith(".png"):
        print("old filename: ", filename)
        old_filename = filename

        y_index = old_filename.index("y")
        new_filename = filename[:(y_index+3)] + "_" + filename[(y_index+3):]
        print("new filename: ", new_filename)

        neg_index = new_filename.index("-")
        new_filename = new_filename[:(neg_index+2)] + "_" + new_filename[(neg_index+2):]

        new_filename = filename.replace("y", "y_")
        new_filename = filename.replace("-", "_-")

        os.rename(old_filename, new_filename)

        print(f"Processed {new_filename}")

print("Finished converting and resizing all images.")