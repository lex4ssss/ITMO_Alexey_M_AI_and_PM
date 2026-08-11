import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poc.generator import MockLLM
from poc.pipeline import Pipeline

HAPPY = "не могу войти в аккаунт, забыл пароль, письмо для сброса не приходит"
BORDERLINE = "Не могу войти, забыл пароль. Почта ivan.petrov@example.com, телефон +7 999 123-45-67"
RISKY = "Требую вернуть деньги за заказ A-104839 на карту 4276 1600 1234 5678, товар не приехал"
LOWCONF = "зелёный кружочек мигает и всё"
OUTAGE = "Сайт не открывается, ошибка 502 уже 20 минут"


def show(title, rec):
    print("=" * 78)
    print(title)
    print("-" * 78)
    for k in ("trace_id", "category", "confidence", "classifier_source", "risk",
              "action", "queue", "reason", "pii_found", "kb_sources", "llm_ok", "latency_ms"):
        print(f"  {k:<18} {rec[k]}")
    print(f"  {'text_redacted':<18} {rec['text_redacted']}")
    if rec["answer"]:
        print("\n  ОТВЕТ ПОЛЬЗОВАТЕЛЮ:\n   ", rec["answer"].replace("\n", "\n    "))
    if rec["draft_for_operator"]:
        print("\n  ЧЕРНОВИК ОПЕРАТОРУ:\n   ", rec["draft_for_operator"].replace("\n", "\n    "))
    print()


def main():
    p = Pipeline()
    show("HAPPY PATH · безопасная категория, высокая уверенность → автоответ",
         p.handle(HAPPY, channel="email"))
    show("BORDERLINE · та же тема, но уверенность ниже порога 0.75 → черновик оператору",
         p.handle(BORDERLINE, channel="email"))
    show("RISKY PATH · возврат денег → всегда оператор, автозакрытие запрещено",
         p.handle(RISKY, channel="chat"))
    show("LOW CONFIDENCE · классификатор не уверен → оператор",
         p.handle(LOWCONF, channel="mobile_app"))
    show("INCIDENT MODE · всплеск обращений о недоступности → присоединение к инциденту",
         p.handle(OUTAGE, channel="web_form", incident_mode=True))

    p_fail = Pipeline(llm=MockLLM(fail=True))
    show("FALLBACK · LLM недоступен на безопасной категории → деградация в оператора",
         p_fail.handle(HAPPY, channel="email"))

    log = Path(__file__).with_name("decisions.jsonl")
    print(f"журнал решений: {log} ({sum(1 for _ in log.open(encoding='utf-8'))} записей)")
    print("последняя запись:")
    print(json.dumps(json.loads(log.read_text(encoding="utf-8").splitlines()[-1]), ensure_ascii=False, indent=2)[:600])


if __name__ == "__main__":
    main()
