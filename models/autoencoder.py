import tensorflow as tf
from tensorflow.keras import layers, Model
import os

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(32, activation="relu")
    ])

    self.decoder = tf.keras.Sequential([
        layers.Dense(128, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dense(80*80*3, activation="sigmoid"),
        layers.Reshape((80,80,3))
    ])

    print("Anomaly Detector created!")

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded