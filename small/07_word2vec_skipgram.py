import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import tensorflow as tf

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
# tokenizer: lowercase, remove punctuation, give every word an id
vec = layers.TextVectorization()
vec.adapt(corpus)
# id 0 = padding, id 1 = unknown word, real words start at id 2
words = [str(w) for w in vec.get_vocabulary()[2:]]
V = vec.vocabulary_size()

# skip-gram data: T = target word, C = context word, L = label
# real neighbour pair (king, queen) -> 1, random wrong pair (king, mango) -> 0
T, C, L = [], [], []
for k, row in enumerate(vec(corpus).numpy()):
    # built-in skipgrams (old Keras function, still inside Keras 3 via tf.keras)
    # window 2 each side, 4 random negative pairs for every real pair
    pairs, labels = tf.keras.preprocessing.sequence.skipgrams(
        row[row > 0], V, window_size=2, negative_samples=4, seed=k
    )
    T += [a for a, _ in pairs]
    C += [b for _, b in pairs]
    L += labels

# 2 embedding tables: one for target words, one for context words
target_emb, context_emb = layers.Embedding(V, 50), layers.Embedding(V, 50)
# 2 inputs: 1 target word id, 1 context word id
ti, ci = keras.Input((1,)), keras.Input((1,))
# Dot = similarity score of the 2 vectors; Flatten -> 1 number per pair
out = layers.Flatten()(layers.Dot(axes=2)([target_emb(ti), context_emb(ci)]))
m = keras.Model([ti, ci], out)
# label is 0/1 -> binary cross-entropy; from_logits=True because no sigmoid at the end
m.compile("adam", keras.losses.BinaryCrossentropy(from_logits=True))
# [:, None] makes shape (N, 1) to match Input((1,))
m.fit(
    [np.array(T)[:, None], np.array(C)[:, None]],
    np.array(L),
    epochs=30,
    batch_size=64,
    verbose=0,
)


# top 3 closest words by cosine similarity (normalize, then dot product)
def similar(E, w):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    # [1:4] skips the word itself (always most similar)
    return [words[i] for i in np.argsort(-E @ E[words.index(w)])[1:4]]


# skip-gram vectors from target table; CBOW vectors from 06 (run 06 first)
sg, cbow = target_emb.get_weights()[0][2:], np.load("cbow.npy")
for w in ["king", "dog", "apple", "python"]:
    print(w, "| skip-gram:", similar(sg, w), "| cbow:", similar(cbow, w))

# CBOW plot
p = PCA(2).fit_transform(cbow)
plt.scatter(*p.T)
for w, xy in zip(words, p):
    plt.annotate(w, xy)
plt.title("CBOW")
plt.show()

# skip-gram plot
p = PCA(2).fit_transform(sg)
plt.scatter(*p.T)
for w, xy in zip(words, p):
    plt.annotate(w, xy)
plt.title("Skip-Gram")
plt.show()
