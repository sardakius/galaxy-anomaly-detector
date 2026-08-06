# Image Control
import os
import glob
import csv
from PIL import Image

# Science 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from scipy.stats import chisquare
from sklearn.metrics import  *
from sklearn.model_selection import train_test_split

# Project Code
from autoencoders import VanillaAnomalyDetector, ConvolutionalAnomalyDetector, ContractiveAnomalyDetector
from model_utils import *

# Zoobot
from pathlib import Path
import pandas as pd
import lightning as L
from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import logout, login


DATASET = "Visu Wavelet"
MODEL = "Contractive Autoencoder"
LOSS = "Gaussian χ2 Error"

DATA_DIR = f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/visu wavelet"
RAW_DATA_DIR = f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/raw"

EPOCHS = 20
BATCH_SIZE = 100

RANDOM_STATE = 1225 # DECEMBER HOLIDAY

if __name__ == "__main__":
    # my model
    normal_images, normal_labels = load_images_from_folder(
        os.path.join(DATA_DIR, 'normal galaxies'), label_value=0
    )
    anomalous_images, anomalous_labels = load_images_from_folder(
        os.path.join(DATA_DIR, 'anomalous galaxies'), label_value=1
    )
    unknown_images, _ = load_images_from_folder(
        os.path.join(DATA_DIR, 'unknown galaxies'), label_value=-99
    )

    unknown_raw_images, _ = load_images_from_folder(
        os.path.join(RAW_DATA_DIR, 'unknown galaxies'), label_value=-99
    )

    # normalize between 0-1
    data = np.array(normal_images + anomalous_images)/255.0
    labels_raw = np.array(normal_labels + anomalous_labels)

    unknown = np.array(unknown_images)/255.0
    unknown_raw = np.array(unknown_raw_images)

    labels = np.eye(2)[labels_raw]

    # split data into training and testing sets, stratifying by labels to maintain class distribution
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_raw
    )

    data_train_normal = data_train[labels_train[:, 0] == 1]


    print(f"Data from the {DATASET} dataset has succesfully been loaded!")

    # create the anomaly detector
    if MODEL == "Regularized Autoencoder":
        anomaly_detector = VanillaAnomalyDetector()
        anomaly_detector.compile(optimizer='adam', loss='mae', metrics=['mae'])
    elif MODEL == "Convolutional Autoencoder":
        anomaly_detector = ConvolutionalAnomalyDetector()
        anomaly_detector.compile(optimizer='adam', loss='mae', metrics=['mae'])
    elif MODEL == "Contractive Autoencoder":
        anomaly_detector = ContractiveAnomalyDetector()
        anomaly_detector.compile(optimizer='adam', loss=anomaly_detector.contractive_loss, metrics=['mae'])
    else:
        print("No valid model found.")
        quit()

    # train it
    history = anomaly_detector.fit(data_train_normal, data_train_normal,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(data_test, data_test)
    )

    print("Training complete!")
    print(f"Evaluating using the {LOSS}...")
    
    # test
    reconstructions = anomaly_detector.predict(data_test)
    # denormalize data
    data_test *= 255
    reconstructions *= 255
    # calculate loss
    test_loss = loss_function(data_test, reconstructions, LOSS)

    # calculate thresholds
    mean = np.mean(test_loss)
    std = np.std(test_loss)

    one_sigma_threshold = mean + 1 * std

    # optimize for best f1 score (check a bunch of thresholds and find the best one)
    thresholds = np.linspace(test_loss.numpy().min(), test_loss.numpy().max(), 10000)

    best_f1 = 0
    best_threshold = 0
    for threshold in thresholds:
        predictions = test_loss > threshold
        test_y_true = labels_test[:, 1]   # 1 = anomaly, 0 = normal
        test_y_pred = predictions.numpy()

        f1 = f1_score(test_y_true, test_y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Best threshold for {LOSS}: {best_threshold:.4f} with F1 score: {best_f1:.4f}")
    print(f"1σ threshold for {LOSS}: {one_sigma_threshold:.4f}")

    # results
    unknown_reconstructions = anomaly_detector.predict(unknown)
    unknown *= 255
    unknown_reconstructions *= 255
    loss = loss_function(unknown, unknown_reconstructions, LOSS)
    predictions = loss > best_threshold
    my_pred = predictions.numpy()

    show_images(
        unknown[my_pred==1],
        unknown[my_pred==1],
        unknown[my_pred==0],
        unknown[my_pred==0],
        title=f"Predictions made by {DATASET}, {MODEL}, {LOSS}",
        figure_path="/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/figures",
        fig_name="visu_wavelet_classifications.png",
        random=True,
        row_names=[f"Anomalous ({len(my_pred[my_pred==1])})", f"Anomalous ({len(my_pred[my_pred==1])})", f"Normal ({len(my_pred[my_pred==0])})", f"Normal ({len(my_pred[my_pred==0])})"]
    )

    show_images(
        unknown_raw[my_pred==1],
        unknown_raw[my_pred==1],
        unknown_raw[my_pred==0],
        unknown_raw[my_pred==0],
        title=f"Predictions made by {DATASET}, {MODEL}, {LOSS}",
        figure_path="/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/figures",
        fig_name="raw_classifications.png",
        random=True,
        row_names=[f"Anomalous ({len(my_pred[my_pred==1])})", f"Anomalous ({len(my_pred[my_pred==1])})", f"Normal ({len(my_pred[my_pred==0])})", f"Normal ({len(my_pred[my_pred==0])})"]
    )
    
    

    print("My model has made its predictions!")


    