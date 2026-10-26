import keras
from keras import layers
import matplotlib.pyplot as plt

(x, y), (xt, yt) = keras.datasets.cifar10.load_data()
x, xt = x / 255, xt / 255

m = keras.Sequential([keras.Input((32, 32, 3))])
for filters in [32, 64, 128]:
    m.add(layers.Conv2D(filters, 3, padding="same", activation="relu"))
    m.add(layers.MaxPooling2D())
m.add(layers.Flatten())
m.add(layers.Dropout(0.5))
m.add(layers.Dense(10, activation="softmax"))

m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
h = m.fit(x, y, epochs=10, batch_size=64, validation_split=0.1).history
print("test acc:", m.evaluate(xt, yt, verbose=0)[1])

for k, v in h.items():
    plt.plot(v, label=k)
plt.legend()
plt.show()
