import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

corpus = [
    "the king rules the kingdom with the queen",
    "the queen is the wife of the king",
    "the prince and princess are children of the king and queen",
    "a man can be a king and a woman can be a queen",
    "the dog chases the cat in the garden",
    "the cat drinks milk and the dog eats meat",
    "dog and cat are pet animals",
    "the cow and horse are farm animals",
    "apple and mango are sweet fruits",
    "she eats an apple and a banana",
    "mango and banana are healthy fruits",
    "python is a programming language",
    "the computer runs python code",
    "deep learning runs on a computer with a gpu",
]
words = sorted(set(" ".join(corpus).split()))
print(words)
idx = {w: i + 1 for i, w in enumerate(words)}
print(idx)
V, W = len(words) + 1, 2

X, Y = [], []
for s in corpus:
    t = [idx[w] for w in s.split()]
    for i in range(len(t)):
        X.append(
            [t[j] if 0 <= j < len(t) else 0 for j in range(i - W, i + W + 1) if j != i]
        )
        Y.append(t[i])

m = keras.Sequential(
    [
        layers.Embedding(V, 50, mask_zero=True),
        layers.GlobalAveragePooling1D(),
        layers.Dense(V, activation="softmax"),
    ]
)
m.compile("adam", "sparse_categorical_crossentropy")
m.fit(np.array(X), np.array(Y), epochs=300, verbose=0)

emb = m.layers[0].get_weights()[0][1:]
np.save("cbow.npy", emb)
p = PCA(2).fit_transform(emb)
plt.scatter(*p.T)
for w, xy in zip(words, p):
    plt.annotate(w, xy)
plt.show()
