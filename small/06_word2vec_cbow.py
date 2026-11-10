import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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


vec = layers.TextVectorization()
vec.adapt(corpus)
words = [str(w) for w in vec.get_vocabulary()[2:]]
V, W = vec.vocabulary_size(), 2

X, Y = [], []
for row in vec(corpus).numpy():
    t = row[row > 0]
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

emb = m.layers[0].get_weights()[0][2:]
np.save("cbow.npy", emb)

p = PCA(2).fit_transform(emb)
plt.scatter(*p.T)
for w, xy in zip(words, p):
    plt.annotate(w, xy)
plt.title("PCA")
plt.show()

p = TSNE(2, perplexity=10, random_state=42).fit_transform(emb)
plt.scatter(*p.T)
for w, xy in zip(words, p):
    plt.annotate(w, xy)
plt.title("t-SNE")
plt.show()
