from time import time
start_time = time() # start the timer

import csv
from anomaly_detector import run_anomaly_detection
from model_utils import secs_to_hms

DATASETS = ["Raw", "Visu Wavelet", "Bayes Wavelet", "Gaussian 0.5σ", "Gaussian 1σ", "Gaussian 2σ", "Noise2Void"]
MODELS = ["Regularized Autoencoder", "Convolutional Autoencoder", "Contractive Autoencoder"]
LOSSES = ["Mean Absolute Error", "χ2 Error", "Gaussian χ2 Error"]

data = []  # list to store results for each combination of dataset, model, and loss function
# run all of the models on all of the datasets with all of the loss functions
for dataset in DATASETS:
    for model in MODELS:
        for loss in LOSSES:
            this_data = run_anomaly_detection(dataset, model, loss)
            data.append(this_data)

# write the results to a CSV file
with open("/Users/ksarthak/Documents/my files/galactic anomaly autoencoder/models/output.csv", mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)

    file.close()
print("All done! Results written to output.csv")
print(f"Total time taken: {secs_to_hms(time() - start_time)}")