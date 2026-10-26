import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np


def prep(img, label):
    img = np.pad(img, ((0, 0), (2, 2), (2, 2)))[..., None] / 255
    return img, (img[..., 0] > 0.5) * (label[:, None, None] + 1)


(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, mx = prep(x[:10000], y[:10000])
xt, mt = prep(xt[:1000], yt[:1000])


def block(z, f):
    z = layers.Conv2D(f, 3, padding="same", activation="relu")(z)
    return layers.Conv2D(f, 3, padding="same", activation="relu")(z)


def down(z):
    return layers.MaxPooling2D()(z)


def up(z):
    return layers.UpSampling2D()(z)


def unet():
    i = keras.Input((32, 32, 1))
    c1 = block(i, 16)
    c2 = block(down(c1), 32)
    c3 = block(down(c2), 64)
    u2 = block(layers.Concatenate()([up(c3), c2]), 32)
    u1 = block(layers.Concatenate()([up(u2), c1]), 16)
    return keras.Model(i, layers.Conv2D(11, 1, activation="softmax")(u1))


def unetpp():
    i = keras.Input((32, 32, 1))
    x00 = block(i, 16)
    x10 = block(down(x00), 32)
    x20 = block(down(x10), 64)
    x01 = block(layers.Concatenate()([x00, up(x10)]), 16)
    x11 = block(layers.Concatenate()([x10, up(x20)]), 32)
    x02 = block(layers.Concatenate()([x00, x01, up(x11)]), 16)
    return keras.Model(i, layers.Conv2D(11, 1, activation="softmax")(x02))


def scores(t, p):
    inter = np.array([((t == c) & (p == c)).sum() for c in range(11)])
    total = np.array([(t == c).sum() + (p == c).sum() for c in range(11)])
    return (t == p).mean(), np.mean(inter / (total - inter)), np.mean(2 * inter / total)


preds = []
for name, build in [("U-Net", unet), ("U-Net++", unetpp)]:
    m = build()
    m.compile("adam", "sparse_categorical_crossentropy")
    m.fit(x, mx, epochs=5, batch_size=64, verbose=0)
    preds.append(m.predict(xt).argmax(-1))
    print(name, "pixel acc=%.4f  IoU=%.4f  Dice=%.4f" % scores(mt, preds[-1]))

plt.imshow(
    np.vstack([np.hstack(a[:8]) for a in [mt, *preds]]), cmap="tab20", vmin=0, vmax=10
)
plt.title("rows: true mask, U-Net, U-Net++")
plt.show()
