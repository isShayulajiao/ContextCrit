"""只在调用方提供的可见证据里检索，不缓存全文。"""


def bigrams(text: str) -> set[str]:
    return {text[i:i + 2] for i in range(len(text) - 1)}


def search(query: str, evidence: list[dict], top_k: int = 5) -> list[dict]:
    query_terms = bigrams(query)
    results = []
    for item in evidence:
        terms = bigrams(item["text"])
        union = query_terms | terms
        score = len(query_terms & terms) / len(union) if union else 0.0
        results.append({**item, "score": score})
    return sorted(results, key=lambda item: (-item["score"], item["evidence_id"]))[:top_k]
