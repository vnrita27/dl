import keras
from keras import layers
import matplotlib.pyplot as plt

(x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
x, xt = x / 255, xt / 255

m = keras.Sequential([layers.Flatten()])
for units in [512, 256, 128]:
    m.add(layers.Dense(units, activation="relu"))
    m.add(layers.BatchNormalization())
    m.add(layers.Dropout(0.3))
m.add(layers.Dense(10, activation="softmax"))

m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
h = m.fit(x, y, epochs=15, batch_size=128, validation_split=0.1).history
print("train acc:", m.evaluate(x, y, verbose=0)[1])
print("test acc :", m.evaluate(xt, yt, verbose=0)[1])

for k, v in h.items():
    plt.plot(v, label=k)
plt.legend()
plt.show()
