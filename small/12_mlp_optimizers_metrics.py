import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = x / 255, xt / 255

for opt in [
    keras.optimizers.SGD(0.01, momentum=0.9),
    keras.optimizers.RMSprop(0.001),
    keras.optimizers.Adam(0.001),
]:
    m = keras.Sequential(
        [
            layers.Flatten(),
            layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    m.compile(opt, "sparse_categorical_crossentropy", metrics=["accuracy"])
    plt.plot(
        m.fit(x, y, epochs=5, validation_split=0.1, verbose=0).history["val_accuracy"],
        label=opt.name,
    )
plt.legend()
plt.show()

m = keras.Sequential(
    [
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ]
)
m.compile("adam", "sparse_categorical_crossentropy")
m.fit(x, y, epochs=5, verbose=0)
p = m.predict(xt)
print(classification_report(yt, p.argmax(1)))
print("ROC AUC:", roc_auc_score(yt, p, multi_class="ovr"))
RocCurveDisplay.from_predictions(keras.utils.to_categorical(yt).ravel(), p.ravel())
plt.show()
