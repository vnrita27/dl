import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = x / 255, xt / 255

m = keras.Sequential([layers.Flatten(), layers.Dense(10, activation="softmax")])
m.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
m.fit(x, keras.utils.to_categorical(y), epochs=10, validation_split=0.1)
print("test acc:", m.evaluate(xt, keras.utils.to_categorical(yt), verbose=0)[1])

ConfusionMatrixDisplay.from_predictions(yt, m.predict(xt).argmax(1))
plt.show()
