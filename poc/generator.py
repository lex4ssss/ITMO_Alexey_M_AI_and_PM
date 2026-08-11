import os
import time


class LLMUnavailable(RuntimeError):
    pass


PROMPT = """Ты оператор поддержки. Ответь пользователю кратко и по делу,
опираясь ТОЛЬКО на фрагменты базы знаний ниже. Если ответа в них нет — так и напиши.

Обращение: {ticket}

Фрагменты базы знаний:
{context}
"""


class MockLLM:
    name = "mock-llm"

    def __init__(self, fail=False, latency_ms=120):
        self.fail = fail
        self.latency_ms = latency_ms

    def generate(self, ticket, chunks):
        time.sleep(self.latency_ms / 1000)
        if self.fail or os.getenv("LLM_FORCE_FAIL") == "1":
            raise LLMUnavailable("LLM API недоступен")
        if not chunks:
            raise LLMUnavailable("нет контекста для ответа")
        head = chunks[0]["text"].split("\n\n")[0].strip()
        body = " ".join(head.split())
        return {
            "text": f"Здравствуйте! {body}\n\nЕсли шаги не помогли, ответьте на это письмо — подключим оператора.",
            "model": self.name,
            "prompt_chars": len(PROMPT.format(ticket=ticket, context="\n".join(c["text"] for c in chunks))),
            "grounded_in": [c["source"] for c in chunks],
        }


def build_llm():
    return MockLLM(fail=os.getenv("LLM_FORCE_FAIL") == "1")
