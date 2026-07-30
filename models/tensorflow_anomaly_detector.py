# TensorFlow Imports
import tensorflow as tf

# Image Control
import os
import glob
from PIL import Image

# Science 
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import chisquare
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Project Code
from autoencoder import AnomalyDetector
from model_utils import load_images_from_folder, show_images, chi_squared_loss

DATASET = "Raw"
MODEL = "Regularized Autoencoder"

DATA_DIR = f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/{DATASET.lower().replace('.', '_')}"
IMG_SIZE = (80, 80)

EPOCHS = 20
BATCH_SIZE = 100

BIN_WIDTH = 0.5

FIGURE_PATH = "tuning/second model/"
# code
if __name__ == "__main__":
    normal_images, normal_labels = load_images_from_folder(
        os.path.join(DATA_DIR, 'normal galaxies'), label_value=0
    )
    anomalous_images, anomalous_labels = load_images_from_folder(
        os.path.join(DATA_DIR, 'anomalous galaxies'), label_value=1
    )

    data_pre_norm = np.array(normal_images + anomalous_images)
    data = data_pre_norm/255.0
    labels_raw = np.array(normal_labels + anomalous_labels)

    labels = np.eye(2)[labels_raw]

    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels_raw
    )

    dpn_train, dpn_test, dpn_train_labels, dpn_test_labels = train_test_split(
        data_pre_norm, labels, test_size=0.2, random_state=42, stratify=labels_raw
    )

    data_train_normal = data_train[labels_train[:, 0] == 1]
    data_train_anomalous = data_train[labels_train[:, 1] == 1]

    data_test_normal = data_test[labels_test[:, 0] == 1]
    data_test_anomalous = data_test[labels_test[:, 1] == 1]

    print("Data has succesfully been loaded!")

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    # model.fit(data_train, labels_train, epochs=20)

    # train_loss, test_acc = model.evaluate(data_train,  labels_train, verbose=2)

    # print('\nTest accuracy:', test_acc)

    # create the anomaly detector
    anomaly_detector = AnomalyDetector()
    anomaly_detector.compile(optimizer='adam', loss='mae', metrics=['mae'])

    history = anomaly_detector.fit(data_train_normal, data_train_normal,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(data_test, data_test))

    normal_reconstructions = anomaly_detector.predict(data_train_normal)
    # denormalize data
    data_train_normal *= 255.0
    normal_reconstructions *= 255.0
    train_loss = chi_squared_loss(data_train_normal, normal_reconstructions) # denormalize data before sendingit in
    
    anomalous_reconstructions = anomaly_detector.predict(data_train_anomalous)
    # anomalous_train_loss = tf.reduce_mean(
    #     tf.abs(anomalous_reconstructions - data_train_anomalous),
    #     axis=(1,2,3)
    # )

    # denormalize data
    data_train_anomalous *= 255.0
    anomalous_reconstructions *= 255.0
    anomalous_train_loss = chi_squared_loss(data_train_anomalous, anomalous_reconstructions)

    reconstructions = anomaly_detector.predict(data_test)
    # test_loss = tf.reduce_mean(
    #     tf.abs(reconstructions - data_test),
    #     axis=(1,2,3)
    # )

    # denormalize data
    data_test *= 255.0
    reconstructions *= 255.0 
    test_loss = chi_squared_loss(data_test, reconstructions)

    print(len(train_loss), len(anomalous_train_loss), len(test_loss))
    mean = np.mean(train_loss)
    std = np.std(train_loss)

    half_sigma_threshold = mean + 0.5 * std
    one_sigma_threshold = mean + 1 * std
    two_sigma_threshold = mean + 2 * std

    plt.figure(figsize=(8,5))
    
    plt.hist(train_loss, bins=int((train_loss.numpy().max() - train_loss.numpy().min())/BIN_WIDTH), alpha=0.6, label="Normal")
    plt.hist(anomalous_train_loss, bins=int((anomalous_train_loss.numpy().max() - anomalous_train_loss.numpy().min())/BIN_WIDTH), alpha=0.6, label="Anomaly")
    plt.xlim(0, np.mean(anomalous_train_loss) + 2*np.std(anomalous_train_loss)) # keep things relatively in order
    plt.axvline(half_sigma_threshold, color="orange", linestyle="--",
                label=f"0.5σ = {half_sigma_threshold:.4f}")
    plt.axvline(one_sigma_threshold, color="red", linestyle="--",
                label=f"1σ = {one_sigma_threshold:.4f}")
    plt.axvline(two_sigma_threshold, color="purple", linestyle="--",
                label=f"2σ = {two_sigma_threshold:.4f}")

    plt.xlabel("Loss")
    plt.ylabel("Number of Images")
    plt.title(f"Distribution of χ² Reconstruction Errors ({DATASET} Dataset, {MODEL})")
    plt.legend()    
    #plt.savefig(f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/figures/{FIGURE_PATH}reconstruction_error_distribution_raw.png")

    threshold = one_sigma_threshold
    predictions = test_loss > threshold
    y_true = labels_test[:, 1]   # 1 = anomaly, 0 = normal
    y_pred = predictions.numpy()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"TP: {tp/len(data_test)*100:.2f}%")
    print(f"TN: {tn/len(data_test)*100:.2f}%")
    print(f"FP: {fp/len(data_test)*100:.2f}%")
    print(f"FN: {fn/len(data_test)*100:.2f}%")

    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]

    false_positive_images = data_test[fp_indices]
    false_negative_images = data_test[fn_indices]
    true_positive_images = data_test[tp_indices]
    true_negative_images = data_test[tn_indices]

    dpn_fp_imgs = dpn_test[fp_indices]
    dpn_fn_imgs = dpn_test[fn_indices]
    dpn_tp_imgs = dpn_test[tp_indices]
    dpn_tn_imgs = dpn_test[tn_indices]

    false_positive_reconstructions = reconstructions[fp_indices]
    false_negative_reconstructions = reconstructions[fn_indices]
    true_positive_reconstructions = reconstructions[tp_indices]
    true_negative_reconstructions = reconstructions[tn_indices]

    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Over Epochs ({DATASET} Dataset, {MODEL})')
    plt.legend()
    #plt.savefig(f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/figures/{FIGURE_PATH}training_validation_loss_raw.png")

    show_images(
        true_positive_images,
        false_positive_images,
        true_negative_images,
        false_negative_images,
        row_names=[f"TP: {tp/len(data_test)*100:.2f}%", f"TN: {tn/len(data_test)*100:.2f}%", f"FP: {fp/len(data_test)*100:.2f}%", f"FN: {fn/len(data_test)*100:.2f}%"],
        title=f"Anomaly Detection Results ({DATASET} Dataset, {MODEL})",
        figure_path=FIGURE_PATH
    )

    # show_images(
    #         dpn_tp_imgs,
    #         dpn_fp_imgs,
    #         dpn_tn_imgs,
    #         dpn_fn_imgs,
    #         row_names=[f"TP: {tp/len(data_test)*100:.2f}%", f"TN: {tn/len(data_test)*100:.2f}%", f"FP: {fp/len(data_test)*100:.2f}%", f"FN: {fn/len(data_test)*100:.2f}%"],
    #         title="Anomaly Detection Results (Adjusted Raw, Pre Normalization Images)",
    #         figure_path=FIGURE_PATH
    #     )
    
    # show_images(
    #     false_positive_images,
    #     false_positive_reconstructions,
    #     false_negative_images,
    #     false_negative_reconstructions,
    #     row_names=["FP Images", "FP Reconstructions", "FN Images", "FN Reconstructions"],
    #     title="False Prediction Image Reconstruction",
    #     figure_path=FIGURE_PATH,
    #     random=False
    # )

    # show_images(
    #     true_positive_images,
    #     true_positive_reconstructions,
    #     true_negative_images,
    #     true_negative_reconstructions,
    #     row_names=["TP Images", "TP Reconstructions", "TN Images", "TN Reconstructions"],
    #     title="True Prediction Image Reconstruction",
    #     figure_path=FIGURE_PATH,
    #     random=False
    # )

    


