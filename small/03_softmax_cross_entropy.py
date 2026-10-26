import keras
from keras import layers
import numpy as np
import tensorflow as tf


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(y, p):
    return -np.sum(y * np.log(p + 1e-12), axis=1)


(x, y), (xt, yt) = keras.datasets.mnist.load_data()
m = keras.Sequential([layers.Flatten(), layers.Dense(10)])
m.compile("adam", keras.losses.CategoricalCrossentropy(from_logits=True))
m.fit(x / 255, keras.utils.to_categorical(y), epochs=1)

logits = m.predict(xt[:5] / 255)
labels = keras.utils.to_categorical(yt[:5], 10)
print("numpy softmax:", softmax(logits)[0].round(4))
print("tf softmax   :", tf.nn.softmax(logits).numpy()[0].round(4))
print("numpy CE:", cross_entropy(labels, softmax(logits)))
print("tf CE   :", tf.nn.softmax_cross_entropy_with_logits(labels, logits).numpy())
print(
    "keras CE (mean):",
    float(keras.losses.CategoricalCrossentropy(from_logits=True)(labels, logits)),
)
