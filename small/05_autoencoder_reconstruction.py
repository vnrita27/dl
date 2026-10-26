import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np

(x, _), (xt, _) = keras.datasets.mnist.load_data()
x, xt = x.reshape(-1, 784) / 255, xt.reshape(-1, 784) / 255

encoder = keras.Sequential(
    [
        keras.Input((784,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(32, activation="relu"),
    ]
)
decoder = keras.Sequential(
    [
        keras.Input((32,)),
        layers.Dense(128, activation="relu"),
        layers.Dense(784, activation="sigmoid"),
    ]
)
ae = keras.Sequential([encoder, decoder])
ae.compile("adam", "binary_crossentropy")
ae.fit(x, x, epochs=10, batch_size=256, validation_data=(xt, xt))

out = ae.predict(xt[:10])
plt.imshow(
    np.vstack(
        [np.hstack(xt[:10].reshape(-1, 28, 28)), np.hstack(out.reshape(-1, 28, 28))]
    ),
    cmap="gray",
)
plt.title("top: original, bottom: reconstructed")
plt.show()
