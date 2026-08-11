from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = Path(__file__).with_name("kb")


def split_chunks(text, max_chars=400):
    parts, cur = [], ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 2 <= max_chars:
            cur = f"{cur}\n\n{para}".strip()
        else:
            if cur:
                parts.append(cur)
            cur = para
    if cur:
        parts.append(cur)
    return parts


class KnowledgeBase:
    def __init__(self, path=KB):
        self.chunks, self.sources = [], []
        for f in sorted(Path(path).glob("*.md")):
            for ch in split_chunks(f.read_text(encoding="utf-8")):
                self.chunks.append(ch)
                self.sources.append(f.name)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.chunks)

    def search(self, query, k=2):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.sources[i], "score": float(sims[i]), "text": self.chunks[i]} for i in order]
