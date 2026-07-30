# TensorFlow Imports
import tensorflow as tf
import torch

# Image Control
import os
import glob
from PIL import Image

# Science 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

IMG_SIZE = (80, 80)

def load_images_from_folder(folder_path, label_value):
    images = []
    labels = []
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    for path in image_paths:
        img = Image.open(path).convert('L')
        img = img.resize(IMG_SIZE)
        img = np.expand_dims(img, axis=-1)
        images.append(np.array(img, dtype=np.float32))
        labels.append(label_value)
    return images, labels

def show_images(tp, fp, tn, fn, title, max_images=7, row_names=['', '', '', ''], figure_path=None, random=True):
    fig, axs = plt.subplots(4, max_images, figsize=(15, 10))

    for i in range(min(max_images, len(tp))):
        if i == 0:
            axs[0, i].set_ylabel(row_names[0], fontsize=12)

        if random:
            img = tp[np.random.randint(0, len(tp))]
        else:
            img = tp[i]

        axs[0, i].imshow(img)

    for i in range(min(max_images, len(fp))):
        if i == 0:
            axs[1, i].set_ylabel(row_names[1], fontsize=12)

        if random:
            img = fp[np.random.randint(0, len(fp))]
        else:
            img = fp[i]
           
        axs[1, i].imshow(img)

    for i in range(min(max_images, len(tn))):
        if i == 0:
            axs[2, i].set_ylabel(row_names[2], fontsize=12)

        if random:
            img = tn[np.random.randint(0, len(tn))]
        else:
            img = tn[i]
           
        axs[2, i].imshow(img)

    for i in range(min(max_images, len(fn))):
        if i == 0:
            axs[3, i].set_ylabel(row_names[3], fontsize=12)

        if random:
            img = fn[np.random.randint(0, len(fn))]
        else:
            img = fn[i]
    
        axs[3, i].imshow(img)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    # plt.savefig(f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/figures/{figure_path}{title.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.show()

def chi_squared_loss(actual, reconstructions):
    actual[actual==0] = 1
    chi_squared = (actual - reconstructions)**2/actual
    return tf.reduce_sum(chi_squared, axis=(1,2,3))