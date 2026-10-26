import keras
from keras import layers
import matplotlib.pyplot as plt

(x, y), _ = keras.datasets.mnist.load_data()
x = x / 255

models = {
    "sigmoid": [
        layers.Dense(256, activation="sigmoid"),
        layers.Dense(128, activation="sigmoid"),
    ],
    "relu": [
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
    ],
    "batchnorm": [
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
    ],
    "dropout": [
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
    ],
}

for name, hidden in models.items():
    keras.utils.set_random_seed(42)
    m = keras.Sequential(
        [layers.Flatten(), *hidden, layers.Dense(10, activation="softmax")]
    )
    m.compile("sgd", "sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(x, y, epochs=10, batch_size=128, validation_split=0.2, verbose=0).history
    print(name, "train:", h["accuracy"][-1], "val:", h["val_accuracy"][-1])
    plt.plot(h["val_accuracy"], label=name)
plt.title("validation accuracy")
plt.legend()
plt.show()
