from gensim.models import Word2Vec
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

url = "https://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
df = pd.read_csv(keras.utils.get_file("quora.tsv", url), sep="\t").fillna("")
y = df.is_duplicate.to_numpy()
q = pd.concat([df.qid1, df.qid2]).value_counts()
print(
    df.shape, "duplicate:", y.mean(), "unique q:", len(q), "repeated q:", (q > 1).sum()
)

q1, q2 = [
    df[c].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    for c in ["question1", "question2"]
]
F = pd.DataFrame(
    {
        "freq1": df.qid1.map(q),
        "freq2": df.qid2.map(q),
        "len1": q1.str.len(),
        "len2": q2.str.len(),
    }
)
F["common"] = [len(set(a.split()) & set(b.split())) for a, b in zip(q1, q2)]
F["word_share"] = F.common / (q1.str.split().str.len() + q2.str.split().str.len() + 1)
for f in [fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio]:
    F[f.__name__] = [f(a, b) for a, b in zip(q1, q2)]

for c in ["word_share", "ratio", "token_sort_ratio", "token_set_ratio"]:
    plt.hist([F[c][y == 0], F[c][y == 1]], 40, density=True, label=["not dup", "dup"])
    plt.title(c)
    plt.legend()
    plt.show()

tr, te = train_test_split(
    np.arange(len(df)), test_size=0.3, stratify=y, random_state=42
)
text = pd.concat([q1[tr], q2[tr]])
tfidf = TfidfVectorizer(tokenizer=str.split, token_pattern=None).fit(text)
w2v = Word2Vec(list(text.str.split()), vector_size=100, min_count=1, seed=42)
emb = w2v.wv[list(tfidf.get_feature_names_out())]


def vec(s):
    W = tfidf.transform(s)
    return W @ emb / np.maximum(W.sum(1).A, 1e-9)


X = np.hstack([F.to_numpy(float), vec(q1), vec(q2)]).astype("float32")

r = np.random.rand(len(te), 2)
probs = {"Random": r / r.sum(1, keepdims=True)}
for name, clf in [
    ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ("Linear SVM", CalibratedClassifierCV(LinearSVC())),
]:
    probs[name] = (
        make_pipeline(StandardScaler(), clf).fit(X[tr], y[tr]).predict_proba(X[te])
    )
for name, p in probs.items():
    print(name, "log loss:", log_loss(y[te], p))
    ConfusionMatrixDisplay.from_predictions(y[te], p.argmax(1), normalize="true")
    plt.title(name)
    plt.show()
