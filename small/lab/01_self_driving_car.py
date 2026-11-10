import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

D = "driving_dataset"
rows = [r.split() for r in open(f"{D}/data.txt") if r.strip()]
paths = np.array([f"{D}/{r[0]}" for r in rows])
y = np.radians([float(r[1].split(",")[0]) for r in rows]).astype("float32")
n1, n2 = int(len(y) * 0.7), int(len(y) * 0.85)
print("frames:", len(y), "mean:", np.degrees(y.mean()), "std:", np.degrees(y.std()))

ks = np.linspace(0, len(y) - 1, 4, dtype=int)
plt.imshow(np.hstack([plt.imread(paths[k]) for k in ks]))
plt.title(f"steering: {np.degrees(y[ks]).round(1)} deg")
plt.show()

plt.hist(np.degrees(y), 100)
plt.title("steering angles")
plt.show()
print("mean baseline test MSE:", np.mean((y[n2:] - y[:n1].mean()) ** 2))


def load(p, a):
    img = tf.io.decode_jpeg(tf.io.read_file(p), channels=3)
    return tf.image.resize(img[-150:], (66, 200)) / 255, a


def batches(a, b):
    return tf.data.Dataset.from_tensor_slices((paths[a:b], y[a:b])).map(load).batch(100)


m = keras.Sequential([keras.Input((66, 200, 3))])
for f, k, s in [(24, 5, 2), (36, 5, 2), (48, 5, 2), (64, 3, 1), (64, 3, 1)]:
    m.add(layers.Conv2D(f, k, strides=s, activation="elu"))
m.add(layers.Flatten())
for u in [100, 50, 10]:
    m.add(layers.Dense(u, activation="elu"))
m.add(layers.Dense(1))
m.compile(keras.optimizers.Adam(1e-4), "mse")
h = m.fit(batches(0, n1).shuffle(100), validation_data=batches(n1, n2), epochs=10)

p = m.predict(batches(n2, None)).ravel()
print("CNN test MSE:", np.mean((y[n2:] - p) ** 2))
plt.plot(h.history["loss"], label="train")
plt.plot(h.history["val_loss"], label="val")
plt.title("loss")
plt.legend()
plt.show()

plt.plot(np.degrees(y[n2:]), label="true")
plt.plot(np.degrees(p), label="predicted")
plt.title("test steering")
plt.legend()
plt.show()
