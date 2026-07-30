import tensorflow as tf
from tensorflow.keras import layers, Model

from IPython import display

class RegularizedAnomalyDetector(Model):
  def __init__(self):
    super(RegularizedAnomalyDetector, self).__init__()
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

class ConvolutionalAnomalyDetector(Model):
  def __init__(self, latent_dim=64):
      super(ConvolutionalAnomalyDetector, self).__init__()
      self.latent_dim = latent_dim

      self.encoder = tf.keras.Sequential([
          layers.InputLayer(shape=(80,80,1)),
          layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
          layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu', padding='same'),
          layers.Flatten(),
          layers.Dense(self.latent_dim)
      ])

      self.decoder = tf.keras.Sequential([
          layers.InputLayer(input_shape=(self.latent_dim,)),
          layers.Dense(20*20*64),
          layers.Reshape((20, 20, 64)),
          tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', activation='relu'),
      ])
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
  # @tf.function
  # def sample(self, eps=None):
  #   if eps is None:
  #     eps = tf.random.normal(shape=(100, self.latent_dim))
  #   return self.decode(eps, apply_sigmoid=True)

  # def encode(self, x):
  #   mean, logvar = tf.split(self.encoder(X), num_or_size_splits=2, axis=1)
  #   return mean, logvar

  # def reparameterize(self, mean, logvar):
  #    eps = tf.random.normal(shape=mean.shape)
  #    return eps * tf.exp(logvar/2) + mean

  # def decode(self, z, apply_sigmoid=False):
  #    logits = self.decoder(z)
  #    if apply_sigmoid:
  #       probs = tf.sigmoid(logits)
  #       return probs
  #    return logits      

  # def fit(epochs=0, batch_size=0, )
  