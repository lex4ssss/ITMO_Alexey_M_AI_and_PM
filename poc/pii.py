import re

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE = re.compile(r"(?:\+7|8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}")

PATTERNS = [("EMAIL", EMAIL), ("CARD", CARD), ("PHONE", PHONE)]


def redact(text):
    found = {}
    out = text
    for name, rx in PATTERNS:
        hits = rx.findall(out)
        if hits:
            found[name] = len(hits)
            out = rx.sub(f"<{name}>", out)
    return out, found


def has_pii(text):
    return bool(redact(text)[1])
