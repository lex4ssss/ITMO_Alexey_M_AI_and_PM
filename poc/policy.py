RISK = {
    "password_reset": "safe",
    "how_to": "safe",
    "payment_failed": "medium",
    "app_bug": "medium",
    "outage_report": "medium",
    "billing_refund": "high",
    "account_deletion": "high",
}

AUTO_CLOSE_THRESHOLD = 0.75
SUGGEST_THRESHOLD = 0.45

QUEUE = {
    "billing_refund": "billing",
    "account_deletion": "compliance",
    "payment_failed": "billing",
    "outage_report": "incident",
    "app_bug": "product",
    "password_reset": "tier1",
    "how_to": "tier1",
}


def risk_of(label):
    return RISK.get(label, "high")


def decide(label, confidence, incident_mode=False, llm_ok=True, pii_found=None):
    risk = risk_of(label)
    queue = QUEUE.get(label, "tier1")
    if risk == "high":
        return {"action": "escalate", "queue": queue, "risk": risk,
                "reason": "категория запрещена к автоматическому закрытию"}
    if incident_mode and label == "outage_report":
        return {"action": "incident_ack", "queue": "incident", "risk": risk,
                "reason": "активен режим инцидента, тикет присоединён к инциденту"}
    if not llm_ok:
        return {"action": "escalate", "queue": queue, "risk": risk,
                "reason": "генерация ответа недоступна, отдаём человеку"}
    if confidence < SUGGEST_THRESHOLD:
        return {"action": "escalate", "queue": queue, "risk": risk,
                "reason": f"низкая уверенность классификатора {confidence:.2f}"}
    if risk == "safe" and confidence >= AUTO_CLOSE_THRESHOLD:
        return {"action": "auto_reply", "queue": queue, "risk": risk,
                "reason": f"безопасная категория, уверенность {confidence:.2f}"}
    return {"action": "suggest", "queue": queue, "risk": risk,
            "reason": f"черновик оператору, уверенность {confidence:.2f}"}
