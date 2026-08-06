# imports
import matplotlib.pyplot as plt
import numpy as np
import csv

DATA_DIR = {
    "Vanilla Autoencoder" : {'roc_auc': [], 'pr_auc': [], 'f1_score': []},
    "Contractive Autoencoder" : {'roc_auc': [], 'pr_auc': [], 'f1_score': []},
    "Convolutional Autoencoder" : {'roc_auc': [], 'pr_auc': [], 'f1_score': []}
}

roc_auc = []
pr_auc = []

with open('output.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        sub_dict = DATA_DIR[row['model']]

        sub_dict['roc_auc'].append(float(row['roc-auc']))
        sub_dict['pr_auc'].append(float(row['pr-auc']))
        sub_dict['f1_score'].append(float(row['f1-score']))

        roc_auc.append(float(row['roc-auc']))
        pr_auc.append(float(row['pr-auc']))


roc_auc = np.array(roc_auc)
pr_auc = np.array(pr_auc)

plt.figure(figsize=(7,7))

# plotting
for (model, sub_dict) in DATA_DIR.items():
    plt.scatter(sub_dict['roc_auc'], sub_dict['pr_auc'], s=np.array(sub_dict['f1_score'])*20, label=model)

plt.xlim(roc_auc.min() - 0.05, roc_auc.max() + 0.05)
plt.ylim(pr_auc.min() - 0.05, pr_auc.max() + 0.05)

# trendline 
z = np.polyfit(roc_auc, pr_auc, 1)
p = np.poly1d(z)

plt.plot(roc_auc, p(roc_auc), "k-", alpha=0.4, label=f"Trendline: y = {z[0]:.2f}x + {z[1]:.2f}")

# labeling
plt.legend()
plt.xlabel("AUC-ROC Score")
plt.ylabel("AUC-PR Score")

plt.title("AUC Scatterplot (Model View)")

# show
plt.show()