import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf

(x, y), _ = keras.datasets.mnist.load_data()
x, y = x[:12800] / 255, y[:12800]
loss_fn = keras.losses.SparseCategoricalCrossentropy()

for clip in [None, 1.0]:
    keras.utils.set_random_seed(42)
    m = keras.Sequential(
        [
            layers.SimpleRNN(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    opt = keras.optimizers.SGD(0.1, momentum=0.9, clipnorm=clip)
    norms, losses = [], []
    for i in range(0, len(x), 64):
        with tf.GradientTape() as tape:
            loss = loss_fn(y[i : i + 64], m(x[i : i + 64]))
        g = tape.gradient(loss, m.trainable_variables)
        opt.apply_gradients(zip(g, m.trainable_variables))
        norms.append(float(tf.linalg.global_norm(g)))
        losses.append(float(loss))
    plt.subplot(1, 2, 1, title="gradient norm", yscale="log").plot(
        norms, label=f"clipnorm={clip}"
    )
    plt.subplot(1, 2, 2, title="loss").plot(losses, label=f"clipnorm={clip}")
plt.legend()
plt.show()
