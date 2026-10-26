import keras
from keras import layers
import matplotlib.pyplot as plt

(x, y), _ = keras.datasets.mnist.load_data()
x = x / 255

for init in ["he_normal", "glorot_uniform", "random_normal"]:
    keras.utils.set_random_seed(42)
    hidden = [
        layers.Dense(128, activation="relu", kernel_initializer=init) for _ in range(5)
    ]
    m = keras.Sequential(
        [layers.Flatten(), *hidden, layers.Dense(10, activation="softmax")]
    )
    m.compile("sgd", "sparse_categorical_crossentropy", metrics=["accuracy"])
    h = m.fit(x, y, epochs=10, batch_size=128, validation_split=0.1, verbose=0).history
    print(init, "val acc:", h["val_accuracy"][-1])
    plt.plot(h["loss"], label=init)
plt.title("training loss")
plt.legend()
plt.show()
