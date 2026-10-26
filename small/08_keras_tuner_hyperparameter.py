import keras
from keras import layers
import keras_tuner as kt

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = x / 255, xt / 255


def build(hp):
    m = keras.Sequential(
        [
            layers.Flatten(),
            layers.Dense(
                hp.Int("units", 32, 512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            ),
            layers.Dense(10, activation="softmax"),
        ]
    )
    m.compile(
        keras.optimizers.Adam(hp.Choice("lr", [1e-2, 1e-3, 1e-4])),
        "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


tuner = kt.RandomSearch(
    build, objective="val_accuracy", max_trials=5, directory="kt", overwrite=True
)
tuner.search(x, y, epochs=3, validation_split=0.2)
print("best:", tuner.get_best_hyperparameters()[0].values)
print("test acc:", tuner.get_best_models()[0].evaluate(xt, yt, verbose=0)[1])
