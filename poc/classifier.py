import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).with_name("data") / "tickets.jsonl"

RULES = [
    ("account_deletion", re.compile(r"удал\w+ (мой )?аккаунт|удал\w+ учётн|стере\w+ данн|персональн\w+ данн", re.I)),
    ("billing_refund", re.compile(r"верн\w+ деньг|возврат средств|возврат оплат|компенсац", re.I)),
    ("outage_report", re.compile(r"не открывается|недоступен|всё лежит|ошибка 50\d|ничего не работает", re.I)),
]


def load(path=DATA):
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r["text"] for r in rows], [r["label"] for r in rows]


def rule_hit(text):
    for label, rx in RULES:
        if rx.search(text):
            return label
    return None


class TicketClassifier:
    def __init__(self):
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, text):
        forced = rule_hit(text)
        proba = self.model.predict_proba([text])[0]
        classes = list(self.model.classes_)
        if forced is not None and forced in classes:
            return {"label": forced, "confidence": max(proba[classes.index(forced)], 0.95), "source": "rule"}
        i = int(proba.argmax())
        return {"label": classes[i], "confidence": float(proba[i]), "source": "model"}


def train(path=DATA):
    X, y = load(path)
    return TicketClassifier().fit(X, y), X, y


def evaluate(path=DATA):
    X, y = load(path)
    clf = TicketClassifier()
    pred = cross_val_predict(clf.model, X, y, cv=5)
    return classification_report(y, pred, digits=3), confusion_matrix(y, pred), sorted(set(y))


if __name__ == "__main__":
    report, cm, labels = evaluate()
    print(report)
    print("метки:", labels)
    print(cm)
