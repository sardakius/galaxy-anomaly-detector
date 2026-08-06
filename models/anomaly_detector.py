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

RANDOM_STATE = 1225 # hahaha deltarune

EPOCHS = 20
BATCH_SIZE = 100

# code
def run_anomaly_detection(DATASET, MODEL, LOSS):
    print(f"Running anomaly detection on the {DATASET} dataset using the {MODEL} and the {LOSS} loss function...")

    DATA_DIR = f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/{DATASET.lower().replace('.', '_')}"
    FIGURE_PATH = f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/figures/{DATASET.lower().replace('.', '_')}/{MODEL.lower().replace(' ', '_')}/{LOSS.lower().replace(' ', '_').replace('χ', 'chi')}"

    os.makedirs(FIGURE_PATH, exist_ok=True)

    normal_images, normal_labels = load_images_from_folder(
        os.path.join(DATA_DIR, 'normal galaxies'), label_value=0
    )
    anomalous_images, anomalous_labels = load_images_from_folder(
        os.path.join(DATA_DIR, 'anomalous galaxies'), label_value=1
    )

    # normalize between 0-1
    data = np.array(normal_images + anomalous_images)/255.0
    labels_raw = np.array(normal_labels + anomalous_labels)

    labels = np.eye(2)[labels_raw]

    # split data into training and testing sets, stratifying by labels to maintain class distribution
    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels_raw
    )

    data_train_normal = data_train[labels_train[:, 0] == 1]
    data_train_anomalous = data_train[labels_train[:, 1] == 1]

    data_test_normal = data_test[labels_test[:, 0] == 1]
    data_test_anomalous = data_test[labels_test[:, 1] == 1]

    print(f"Data from the {DATASET} dataset has succesfully been loaded!")

    # create the anomaly detector
    if MODEL == "Vanilla Autoencoder":
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
    predictions = test_loss > best_threshold
    y_true = labels_test[:, 1]   #1 = anomaly, 0 = normal
    y_pred = predictions.numpy()

    normal_val = test_loss[y_true == 0]
    anomalous_val = test_loss[y_true == 1]

    # # distribution of errors
    # plt.figure(figsize=(9, 6))
    # plt.hist(normal_val, bins=int(np.sqrt(len(normal_val))), alpha=0.6, label="Normal")
    # plt.hist(anomalous_val, bins=int(np.sqrt(len(anomalous_val))), alpha=0.6, label="Anomalous")
    # plt.axvline(one_sigma_threshold, color="red", linestyle="--", label=f"1σ = {one_sigma_threshold:.4f}")
    # plt.axvline(best_threshold, color="green", linestyle="--", label=f"Threshold = {best_threshold:.4f}")
    # plt.xlabel("Reconstruction Error")   
    # plt.ylabel("Number of Images")
    # plt.title(f"Distribution of Reconstruction Errors ({DATASET}, {MODEL}, {LOSS})", wrap=True)
    # plt.legend()  
    # plt.savefig(os.path.join(FIGURE_PATH, "reconstruction_error_distribution.png"))

    # confusing the matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"TP: {tp} ({tp/len(data_test)*100:.2f}%)")
    print(f"TN: {tn} ({tn/len(data_test)*100:.2f}%)")
    print(f"FP: {fp} ({fp/len(data_test)*100:.2f}%)")
    print(f"FN: {fn} ({fn/len(data_test)*100:.2f}%)")

    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]

    false_positive_images = data_test[fp_indices]
    false_negative_images = data_test[fn_indices]
    true_positive_images = data_test[tp_indices]
    true_negative_images = data_test[tn_indices]

    false_positive_reconstructions = reconstructions[fp_indices]
    false_negative_reconstructions = reconstructions[fn_indices]
    true_positive_reconstructions = reconstructions[tp_indices]
    true_negative_reconstructions = reconstructions[tn_indices]

    # # Plotting Confusion Matrix
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomalous"])
    # disp.plot(cmap=plt.cm.Blues, values_format='d')
    # plt.title(f"Confusion Matrix ({DATASET} Dataset, {MODEL}, {LOSS} Function)", wrap=True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(FIGURE_PATH, "confusion_matrix.png"))

    # # plotting training and validation loss over epochs
    # plt.figure(figsize=(9, 5))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss (Mean Absolute Error)')
    # plt.title(f'Loss Over Epochs ({DATASET}, {MODEL})')
    # plt.legend()
    # plt.savefig(os.path.join(FIGURE_PATH, "training_validation_loss.png"))

    # Results
    show_images(
        true_positive_images,
        true_negative_images,
        false_positive_images,
        false_negative_images,
        row_names=[f"TP: {tp/len(data_test)*100:.2f}%", f"TN: {tn/len(data_test)*100:.2f}%", f"FP: {fp/len(data_test)*100:.2f}%", f"FN: {fn/len(data_test)*100:.2f}%"],
        title=f"Anomaly Detection Results ({DATASET} Dataset, {MODEL}, {LOSS} Function)",
        fig_name='anomaly_detector_Results.png',
        figure_path=FIGURE_PATH
    )

    # # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, test_loss)
    pr_auc = auc(recall, precision)
    print(f"Precision-Recall AUC: {pr_auc}")

    # plt.figure(figsize=(8, 8))
    # plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title(f'Precision-Recall Curve ({DATASET} Dataset, {MODEL}, {LOSS} Function)', wrap=True)
    # plt.legend()
    # plt.savefig(os.path.join(FIGURE_PATH, "precision_recall_curve.png"))    

    # # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, test_loss)
    roc_auc = auc(fpr, tpr)
    print(f"ROC AUC: {roc_auc}")

    # plt.figure(figsize=(8, 8))
    # plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    # plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'Reciever Operating Curve ({DATASET} Dataset, {MODEL}, {LOSS} Function)',wrap=True)
    # plt.legend()
    # plt.savefig(os.path.join(FIGURE_PATH, "roc_curve.png"))

    plt.close('all')  # Close all figures to free memory

    return {
        "dataset": DATASET,
        "model": MODEL,
        "loss": LOSS,
        "pr-auc": f"{pr_auc:.4f}",
        "roc-auc": f"{roc_auc:.4f}",
        "f1-score": f"{best_f1:.4f}"
    }