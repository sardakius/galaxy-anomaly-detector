import tensorflow as tf
from tensorflow.keras import layers, Model, backend as K

from IPython import display

class RegularizedAnomalyDetector(Model):
  def __init__(self):
    super(RegularizedAnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(1024, activation="relu"),
      layers.Dense(512, activation="relu"),
      layers.Dense(128, activation="relu")
    ])

    self.decoder = tf.keras.Sequential([
      layers.Dense(512, activation="relu"),
      layers.Dense(1024, activation="relu"),
      layers.Dense(80*80*1, activation="sigmoid"),
      layers.Reshape((80, 80, 1))
    ])

    print("Regularized anomaly detector created!")

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class ConvolutionalAnomalyDetector(Model):
  def __init__(self, latent_dim=4):
      super(ConvolutionalAnomalyDetector, self).__init__()
      self.latent_dim = latent_dim

      self.encoder = tf.keras.Sequential([
          layers.InputLayer(shape=(80,80,1)),
          layers.Flatten(),
          layers.Dense(self.latent_dim)
      ])

      self.decoder = tf.keras.Sequential([
          layers.InputLayer(shape=(self.latent_dim,)),
          layers.Dense(80*80*1),
          layers.Reshape((80, 80, 1))
      ])

      print("Convolutional anomaly detector created!")

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class ContractiveAnomalyDetector(Model):
  def __init__(self):
    super(ContractiveAnomalyDetector, self).__init__()

    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(32, activation='relu'),
      layers.Dense(16, activation='relu')
    ])

    self.h = 0

    self.decoder = tf.keras.Sequential([
      layers.Dense(32, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(80*80*1, activation='relu'),
      layers.Reshape((80, 80, 1))
    ])

    print("Contractive anomaly detector created!")

  def call(self, x):
    encoded = self.encoder(x)
    self.h = encoded
    decoded = self.decoder(encoded)
    return decoded

  def contractive_loss(self, y_true, y_pred):
      lm = 1e-4
      mae = K.mean(K.abs(y_true - y_pred), axis=1)
  
      weights = self.get_layer('sequential_1').layers[-1].kernel
      weights = K.transpose(weights)
  
      penalty_term =  K.sum(((self.h * (1 - self.h))**2) * K.sum(weights**2, axis=1), axis=1)
  
      loss = mae + (lm * penalty_term)
  
      return loss