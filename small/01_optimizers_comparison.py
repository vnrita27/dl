import keras
from keras import layers
import matplotlib.pyplot as plt

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = x / 255, xt / 255

for opt in ["sgd", "adagrad", "rmsprop", "adam"]:
    keras.utils.set_random_seed(42)
    m = keras.Sequential(
        [
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    m.compile(opt, "sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(x, y, epochs=5, batch_size=128, validation_split=0.1, verbose=0).history
    print(opt, "test acc:", m.evaluate(xt, yt, verbose=0)[1])
    plt.subplot(1, 2, 1, title="loss").plot(h["loss"], label=opt)
    plt.subplot(1, 2, 2, title="val accuracy").plot(h["val_accuracy"], label=opt)
plt.legend()
plt.show()
