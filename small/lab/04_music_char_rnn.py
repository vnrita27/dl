import keras
from keras import layers
import numpy as np

TABLA = False
base = (
    "https://raw.githubusercontent.com/jukedeck/nottingham-dataset/master/ABC_cleaned/"
)
files = ["jigs.abc", "hpps.abc", "reelsa-c.abc"]
text = (
    open("tabla.txt").read()
    if TABLA
    else "".join(open(keras.utils.get_file(f, base + f)).read() for f in files)
)
chars = sorted(set(text))
idx = {c: i for i, c in enumerate(chars)}
ids = np.array([idx[c] for c in text])
print(len(text), "chars,", len(chars), "unique")

B, T = 16, 64
L = (len(ids) - 1) // (B * T) * B * T


def cut(a):
    return a.reshape(B, -1, T).swapaxes(0, 1).reshape(-1, T)


X, Y = cut(ids[:L]), cut(ids[1 : L + 1])


def build(b, t):
    return keras.Sequential(
        [
            keras.Input(batch_shape=(b, t)),
            layers.Embedding(len(chars), 256),
            layers.LSTM(256, return_sequences=True, stateful=True),
            layers.LSTM(256, return_sequences=True, stateful=True),
            layers.TimeDistributed(layers.Dense(len(chars), activation="softmax")),
        ]
    )


m = build(B, T)
m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
m.summary()
for epoch in range(30):
    for ly in m.layers[1:3]:
        ly.reset_state()
    m.fit(X, Y, batch_size=B, shuffle=False, verbose=2)

g = build(1, 1)
g.set_weights(m.get_weights())
out = c = text[0]
for _ in range(800):
    p = g(np.array([[idx[c]]])).numpy()[0, -1].astype("float64")
    c = np.random.choice(chars, p=p / p.sum())
    out += c
print(out)
