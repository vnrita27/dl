import os
import zipfile

import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import LinearSVC

url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
root = keras.utils.get_file("har.zip", url, extract=True)
D = f"{root}/UCI HAR Dataset"
if not os.path.isdir(D):
    zipfile.ZipFile(D + ".zip").extractall(root)


def read(f, **kw):
    return np.loadtxt(f"{D}/{f}", **kw)


acts = read("activity_labels.txt", dtype=str, usecols=1)
X, Xt = read("train/X_train.txt"), read("test/X_test.txt")
y, yt = read("train/y_train.txt", dtype=int) - 1, read("test/y_test.txt", dtype=int) - 1
print(
    "train",
    X.shape,
    "test",
    Xt.shape,
    "missing:",
    np.isnan(X).sum(),
    "duplicates:",
    len(X) - len(np.unique(X, axis=0)),
)

plt.bar(acts, np.bincount(y))
plt.xticks(rotation=45)
plt.title("activity counts")
plt.show()

for i, a in enumerate(acts):
    plt.hist(X[y == i, 200], 50, alpha=0.5, label=a)
plt.title("tBodyAccMag-mean")
plt.legend()
plt.show()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(200),
}
for name, clf in models.items():
    print(name, "test acc:", clf.fit(X, y).score(Xt, yt))
ConfusionMatrixDisplay.from_estimator(
    models["Linear SVM"], Xt, yt, display_labels=acts, xticks_rotation=45
)
plt.show()


def signals(s):
    return np.stack(
        [
            read(f"{s}/Inertial Signals/{t}_{a}_{s}.txt")
            for t in ["body_acc", "body_gyro", "total_acc"]
            for a in "xyz"
        ],
        -1,
    )


S, St = signals("train"), signals("test")
S, St = (S - S.mean((0, 1))) / S.std((0, 1)), (St - S.mean((0, 1))) / S.std((0, 1))

lstm = keras.Sequential(
    [layers.LSTM(64), layers.Dropout(0.5), layers.Dense(6, activation="softmax")]
)
cnn = keras.Sequential(
    [
        layers.Conv1D(64, 5, activation="relu"),
        layers.MaxPooling1D(),
        layers.Conv1D(64, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(6, activation="softmax"),
    ]
)
for name, m in [("LSTM", lstm), ("1D CNN", cnn)]:
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(S, y, epochs=30, batch_size=64, verbose=2)
    print(name, "test acc:", m.evaluate(St, yt, verbose=0)[1])
