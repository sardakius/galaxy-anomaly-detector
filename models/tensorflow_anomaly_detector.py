# Scikit Imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# TensorFlow Imports
import tensorflow as tf
from tensorflow import keras

# Autoencoder
from autoencoder import AnomalyDetector

# Python
import os
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/galaxy data/raw"
IMG_SIZE = (80, 80)

# code
def load_images_from_folder(folder_path, label_value):
    images = []
    labels = []
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize(IMG_SIZE)
        images.append(np.array(img, dtype=np.float32))
        labels.append(label_value)
    return images, labels

def show_images(tp, fp, tn, fn, title, max_images=7):
    fig, axs = plt.subplots(4, max_images, figsize=(15, 10))

    for i in range(min(max_images, len(tp))):
        if i == 0:
            axs[0, i].set_ylabel(f'TP ({len(tp)/len(data_test)*100:.2f}%)', fontsize=12)
        axs[0, i].imshow(tp[i])

    for i in range(min(max_images, len(fp))):
        if i == 0:
            axs[1, i].set_ylabel(f'FP ({len(fp)/len(data_test)*100:.2f}%)', fontsize=12)
        axs[1, i].imshow(fp[i])

    for i in range(min(max_images, len(tn))):
        if i == 0:
            axs[2, i].set_ylabel(f'TN ({len(tn)/len(data_test)*100:.2f}%)', fontsize=12)
        axs[2, i].imshow(tn[i])

    for i in range(min(max_images, len(fn))):
        if i == 0:
            axs[3, i].set_ylabel(f'FN ({len(fn)/len(data_test)*100:.2f}%)', fontsize=12)
        axs[3, i].imshow(fn[i])

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f"/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/figures/{title.replace(' ', '_').replace('(', '').replace(')', '')}.png")
    plt.show()

if __name__ == "__main__":
    normal_images, normal_labels = load_images_from_folder(
        os.path.join(DATA_DIR, 'normal galaxies'), label_value=0
    )
    anomalous_images, anomalous_labels = load_images_from_folder(
        os.path.join(DATA_DIR, 'anomalous galaxies'), label_value=1
    )

    data = np.array(normal_images + anomalous_images) / 255.0
    labels_raw = np.array(normal_labels + anomalous_labels)

    labels = np.eye(2)[labels_raw]

    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels_raw
    )

    data_train_normal = data_train[labels_train[:, 0] == 1]
    data_train_anomalous = data_train[labels_train[:, 1] == 1]

    data_test_normal = data_test[labels_test[:, 0] == 1]
    data_test_anomalous = data_test[labels_test[:, 1] == 1]

    print("Data has succesfully been loaded!")

    # create the anomaly detector
    anomaly_detector = AnomalyDetector()
    anomaly_detector.compile(optimizer='adam', loss='mae', metrics=['mae'])
    
    history = anomaly_detector.fit(data_train_normal, data_train_normal,
            epochs=20,
            batch_size=100,
            validation_data=(data_test, data_test),
            shuffle=True)

    reconstructions = anomaly_detector.predict(data_test)
    test_loss = np.mean(np.abs(reconstructions - data_test), axis=(1,2,3))

    normal_errors = test_loss[labels_test[:, 0] == 1]
    anomaly_errors = test_loss[labels_test[:, 1] == 1]

    train_reconstructions = anomaly_detector.predict(data_train_normal)
    train_loss = np.mean(
        np.abs(train_reconstructions - data_train_normal),
        axis=(1,2,3)
    )
    half_sigma = np.mean(train_loss) + 0.5 * np.std(train_loss)
    one_sigma = np.mean(train_loss) + np.std(train_loss) # FOR NOW, select a threshold that is one std avove the mean
    two_sigma = np.mean(train_loss) + 2 * np.std(train_loss)

    plt.figure(figsize=(8,5))

    plt.hist(normal_errors, bins=40, alpha=0.6, label="Normal")
    plt.hist(anomaly_errors, bins=40, alpha=0.6, label="Anomaly")


    plt.axvline(one_sigma, color="red", linestyle="--",
                label=f"1σ = {one_sigma:.4f}")
    plt.axvline(half_sigma, color="orange", linestyle="--",
                label=f"0.5σ = {half_sigma:.4f}")
    plt.axvline(two_sigma, color="purple", linestyle="--",
                label=f"2σ = {two_sigma:.4f}")

    plt.xlabel("Reconstruction Error (MAE)")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Reconstruction Errors (Raw Dataset)")
    plt.legend()
    plt.show()
    plt.savefig("/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/figures/reconstruction_error_distribution_raw.png")

    threshold = two_sigma
    predictions = test_loss > threshold
    y_true = labels_test[:, 1]   # 1 = anomaly, 0 = normal
    y_pred = predictions.astype(int)

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

    show_images(
        true_positive_images,
        false_positive_images,
        true_negative_images,
        false_negative_images,
        title="Anomaly Detection Results (Raw)"
    )

