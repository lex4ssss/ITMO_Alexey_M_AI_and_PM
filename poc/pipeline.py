import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import policy
from .classifier import train
from .generator import LLMUnavailable, build_llm
from .pii import redact
from .retriever import KnowledgeBase

LOG = Path(__file__).with_name("decisions.jsonl")


class Pipeline:
    def __init__(self, llm=None, log_path=LOG):
        self.clf, _, _ = train()
        self.kb = KnowledgeBase()
        self.llm = llm or build_llm()
        self.log_path = Path(log_path)

    def handle(self, ticket_text, channel="chat", incident_mode=False):
        t0 = time.perf_counter()
        trace = str(uuid.uuid4())[:8]

        clean, pii_found = redact(ticket_text)
        cls = self.clf.predict(clean)
        t_class = (time.perf_counter() - t0) * 1000

        chunks = self.kb.search(clean, k=2)
        draft, llm_ok, llm_error = None, True, None
        card_seen = "CARD" in pii_found
        decision = policy.decide(cls["label"], cls["confidence"], incident_mode, True, card_seen)

        if decision["action"] in ("auto_reply", "suggest"):
            try:
                draft = self.llm.generate(clean, chunks)
            except LLMUnavailable as e:
                llm_ok, llm_error = False, str(e)
                decision = policy.decide(cls["label"], cls["confidence"], incident_mode, False, card_seen)

        total = (time.perf_counter() - t0) * 1000
        record = {
            "trace_id": trace,
            "ts": datetime.now(timezone.utc).isoformat(),
            "channel": channel,
            "text_redacted": clean,
            "pii_found": pii_found,
            "category": cls["label"],
            "confidence": round(cls["confidence"], 4),
            "classifier_source": cls["source"],
            "risk": decision["risk"],
            "action": decision["action"],
            "queue": decision["queue"],
            "reason": decision["reason"],
            "kb_sources": [c["source"] for c in chunks],
            "kb_top_score": round(chunks[0]["score"], 4) if chunks else 0.0,
            "llm_ok": llm_ok,
            "llm_error": llm_error,
            "answer": draft["text"] if draft and decision["action"] == "auto_reply" else None,
            "draft_for_operator": draft["text"] if draft and decision["action"] == "suggest" else None,
            "latency_ms": {"classify": round(t_class, 1), "total": round(total, 1)},
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
