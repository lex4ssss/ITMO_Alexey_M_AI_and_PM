import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poc.classifier import train
from poc.generator import MockLLM
from poc.pii import redact
from poc.pipeline import Pipeline
from poc.policy import decide
from poc.retriever import KnowledgeBase

PIPE = Pipeline()


def test_pii_redacted():
    text = "почта a@b.com, телефон +7 999 123-45-67, карта 4276 1600 1234 5678"
    clean, found = redact(text)
    assert "a@b.com" not in clean
    assert "4276" not in clean
    assert set(found) == {"EMAIL", "PHONE", "CARD"}


def test_pii_not_in_log():
    rec = PIPE.handle("сбросьте пароль, почта ivan.petrov@example.com")
    assert "ivan.petrov@example.com" not in rec["text_redacted"]
    assert rec["pii_found"].get("EMAIL") == 1


def test_refund_never_auto():
    rec = PIPE.handle("верните деньги за заказ A-104839, товар не приехал")
    assert rec["category"] == "billing_refund"
    assert rec["action"] == "escalate"
    assert rec["answer"] is None


def test_deletion_never_auto():
    rec = PIPE.handle("удалите мой аккаунт и все персональные данные")
    assert rec["risk"] == "high"
    assert rec["action"] == "escalate"


def test_happy_path_auto_reply():
    rec = PIPE.handle("не могу войти, забыл пароль, письмо для сброса не приходит")
    assert rec["category"] == "password_reset"
    assert rec["action"] == "auto_reply"
    assert rec["answer"]


def test_llm_failure_degrades_to_operator():
    rec = Pipeline(llm=MockLLM(fail=True)).handle("забыл пароль, не приходит письмо")
    assert rec["llm_ok"] is False
    assert rec["action"] == "escalate"
    assert rec["answer"] is None


def test_low_confidence_escalates():
    d = decide("app_bug", 0.20)
    assert d["action"] == "escalate"


def test_incident_mode_acks():
    rec = PIPE.handle("сайт не открывается, ошибка 502", incident_mode=True)
    assert rec["action"] == "incident_ack"
    assert rec["queue"] == "incident"


def test_classifier_quality():
    clf, X, y = train()
    hits = sum(clf.predict(t)["label"] == lab for t, lab in zip(X, y))
    assert hits / len(y) > 0.9


def test_kb_returns_sources():
    hits = KnowledgeBase().search("забыл пароль, письмо не приходит", k=2)
    assert hits and hits[0]["score"] > 0
    assert hits[0]["source"].endswith(".md")


def test_latency_budget():
    rec = PIPE.handle("как изменить адрес доставки")
    assert rec["latency_ms"]["classify"] < 500
