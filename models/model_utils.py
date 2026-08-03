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


def load_images_from_folder(folder_path, label_value):
    images = []
    labels = []
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    for path in image_paths:
        img = Image.open(path).convert('L')
        img = img.resize((80, 80))
        img = np.expand_dims(img, axis=-1)
        images.append(np.array(img, dtype=np.float32))
        labels.append(label_value)
    return images, labels

def show_images(tp, fp, tn, fn, title, max_images=7, row_names=['', '', '', ''], figure_path=None, random=False):
    fig, axs = plt.subplots(4, max_images, figsize=(15, 10))

    for i in range(0, 4):
        axs[i, 0].set_ylabel(row_names[i], fontsize=12)

    for i in range(min(max_images, len(tp))):
        if random:
            img = tp[np.random.randint(0, len(tp))]
        else:
            img = tp[i]
        axs[0, i].imshow(img)

    for i in range(min(max_images, len(fp))):
        if random:
            img = fp[np.random.randint(0, len(fp))]
        else:
            img = fp[i]
        axs[1, i].imshow(img)

    for i in range(min(max_images, len(tn))):
        if random:
            img = tn[np.random.randint(0, len(tn))]
        else:
            img = tn[i]
        axs[2, i].imshow(img)

    for i in range(min(max_images, len(fn))):
        if random:
            img = fn[np.random.randint(0, len(fn))]
        else:
            img = fn[i] 
        axs[3, i].imshow(img)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"))

def loss_function(actual, reconstructions, loss_type):
    if loss_type == "Mean Absolute Error":
        return mae(actual, reconstructions)
    elif loss_type == "χ2 Error":
        return chi_squared_loss(actual, reconstructions)
    elif loss_type == "Gaussian χ2 Error":
        return gaussian_chi_squared_loss(actual, reconstructions)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}. Supported types are: 'regular', 'chi_squared', 'sigma_clipped_chi_squared', 'gaussian_chi_squared'.")

def mae(actual, reconstructions):
    return tf.reduce_mean(tf.abs(actual - reconstructions), axis=(1,2,3))

def chi_squared_loss(actual, reconstructions):
    actual_copy = actual.copy()
    actual_copy[actual_copy==0] = 1
    
    chi_squared = tf.abs((actual - reconstructions)**2)/actual_copy

    return tf.reduce_mean(chi_squared, axis=(1,2,3))

def create_gaussian(shape, sigma=7.5):
    grid_size = shape[1]
    center = (grid_size - 1) / 2.0

    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y, indexing='ij')
    gaussian = np.exp(-((X - center) ** 2 + (Y - center) ** 2) / (2 * sigma ** 2))
    gaussian = gaussian/gaussian.max() 

    gaussian = gaussian[np.newaxis, :, :, np.newaxis]

    return gaussian

def gaussian_chi_squared_loss(actual, reconstructions):
    actual_copy = actual.copy()
    actual_copy[actual_copy==0] = 1

    chi_squared = tf.abs((actual - reconstructions)**2)/actual_copy
        
    # gaussian weighed mask
    gaussian = create_gaussian(actual.shape)

    return tf.reduce_mean(chi_squared*gaussian, axis=(1,2,3))