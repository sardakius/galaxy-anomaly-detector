import tensorflow as tf
from tensorflow.keras import layers, Model
import os

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(512, activation="relu"),
      layers.Dense(256, activation="relu"),
      layers.Dense(64, activation="relu")
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(256, activation="relu"),
      layers.Dense(512, activation="relu"),
      layers.Dense(80*80*1, activation="sigmoid"),
      layers.Reshape((80, 80, 1))
    ])

    print("Anomaly Detector created!")

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded