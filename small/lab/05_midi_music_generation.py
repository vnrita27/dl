import keras
from keras import layers
from music21 import corpus
from music21 import note
from music21 import stream
import numpy as np

seq = [
    f"{n.pitches[-1].nameWithOctave} {float(n.quarterLength)}"
    for p in corpus.getComposer("bach")[:100]
    for n in corpus.parse(p).parts[0].flatten().notes
]
vocab = sorted(set(seq))
ids = np.array([vocab.index(t) for t in seq])
print(len(seq), "notes,", len(vocab), "unique")

T = 32
X = np.array([ids[i : i + T] for i in range(len(ids) - T)])
m = keras.Sequential(
    [
        layers.Embedding(len(vocab), 64),
        layers.LSTM(256, return_sequences=True),
        layers.LSTM(256),
        layers.Dropout(0.3),
        layers.Dense(len(vocab), activation="softmax"),
    ]
)
m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
m.fit(X, ids[T:], epochs=30, batch_size=64, validation_split=0.1)

w, song = list(X[0]), stream.Stream()
for _ in range(100):
    p = m.predict(np.array([w[-T:]]), verbose=0)[0].astype("float64")
    w.append(np.random.choice(len(vocab), p=p / p.sum()))
    pitch, dur = vocab[w[-1]].split()
    song.append(note.Note(pitch, quarterLength=float(dur)))
song.write("midi", "generated.mid")
