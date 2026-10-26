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
idx = {w: i + 1 for i, w in enumerate(words)}
V, W = len(words) + 1, 2

T, C, L = [], [], []
for s in corpus:
    t = [idx[w] for w in s.split()]
    for i in range(len(t)):
        for j in range(max(0, i - W), min(len(t), i + W + 1)):
            if i != j:
                T += [t[i]] * 5
                C += [t[j], *np.random.randint(1, V, 4)]
                L += [1, 0, 0, 0, 0]

target_emb, context_emb = layers.Embedding(V, 50), layers.Embedding(V, 50)
ti, ci = keras.Input((1,)), keras.Input((1,))
out = layers.Flatten()(layers.Dot(axes=2)([target_emb(ti), context_emb(ci)]))
m = keras.Model([ti, ci], out)
m.compile("adam", keras.losses.BinaryCrossentropy(from_logits=True))
m.fit(
    [np.array(T)[:, None], np.array(C)[:, None]],
    np.array(L),
    epochs=30,
    batch_size=64,
    verbose=0,
)


def similar(E, w):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return [words[i] for i in np.argsort(-E @ E[words.index(w)])[1:4]]


sg, cbow = target_emb.get_weights()[0][1:], np.load("cbow.npy")
for w in ["king", "dog", "apple", "python"]:
    print(w, "| skip-gram:", similar(sg, w), "| cbow:", similar(cbow, w))

for i, (name, E) in enumerate([("CBOW", cbow), ("Skip-Gram", sg)]):
    p = PCA(2).fit_transform(E)
    ax = plt.subplot(1, 2, i + 1, title=name)
    ax.scatter(*p.T)
    for w, xy in zip(words, p):
        ax.annotate(w, xy)
plt.show()
