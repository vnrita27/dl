import keras
from keras import layers
import matplotlib.pyplot as plt

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = x / 255, xt / 255

H = {}
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
    H[opt] = m.fit(
        x, y, epochs=5, batch_size=128, validation_split=0.1, verbose=0
    ).history
    print(opt, "test acc:", m.evaluate(xt, yt, verbose=0)[1])

for opt in H:
    plt.plot(H[opt]["loss"], label=opt)
plt.title("loss")
plt.legend()
plt.show()

for opt in H:
    plt.plot(H[opt]["val_accuracy"], label=opt)
plt.title("val accuracy")
plt.legend()
plt.show()
