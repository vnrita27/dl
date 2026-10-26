import keras
from keras import layers
import matplotlib.pyplot as plt

(x, y), (xt, yt) = keras.datasets.cifar10.load_data()
x, y, xt, yt = x[:5000], y[:5000], xt[:1000], yt[:1000]

nets = {
    "ResNet50": (
        keras.applications.ResNet50,
        keras.applications.resnet50.preprocess_input,
    ),
    "MobileNetV3Small": (
        keras.applications.MobileNetV3Small,
        keras.applications.mobilenet_v3.preprocess_input,
    ),
    "EfficientNetV2B0": (
        keras.applications.EfficientNetV2B0,
        keras.applications.efficientnet_v2.preprocess_input,
    ),
}
acc = {}
for name, (net, preprocess) in nets.items():
    base = net(
        include_top=False, weights="imagenet", input_shape=(96, 96, 3), pooling="avg"
    )
    base.trainable = False
    m = keras.Sequential(
        [
            keras.Input((32, 32, 3)),
            layers.Resizing(96, 96),
            layers.Lambda(preprocess),
            base,
            layers.Dense(10, activation="softmax"),
        ]
    )
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(x, y, epochs=3, batch_size=64, verbose=0)
    acc[name] = m.evaluate(xt, yt, verbose=0)[1]
    print(f"{name}: test acc={acc[name]:.4f}  params={base.count_params():,}")

plt.bar(acc.keys(), acc.values())
plt.title("pretrained backbone accuracy on CIFAR-10")
plt.show()
