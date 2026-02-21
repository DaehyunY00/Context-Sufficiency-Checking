from __future__ import annotations

import re
from typing import Dict, List, Set

from sufficiency.base import BaseChecker, INSUFFICIENT, SUFFICIENT


_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "to",
    "of",
    "in",
    "on",
    "for",
    "and",
    "or",
    "by",
    "with",
    "from",
    "at",
    "who",
    "what",
    "when",
    "where",
    "which",
    "why",
    "how",
    "do",
    "does",
    "did",
    "be",
    "have",
    "has",
    "had",
}


class KeywordCoverageChecker(BaseChecker):
    """질문 키워드 커버리지 기반 충분성 판별기."""

    name = "heuristic"

    def __init__(self, min_keyword_hits: int = 2, min_coverage_ratio: float = 0.35) -> None:
        self.min_keyword_hits = int(min_keyword_hits)
        self.min_coverage_ratio = float(min_coverage_ratio)

    def _extract_keywords(self, question: str) -> List[str]:
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", question.lower())
        keywords = [w for w in candidates if w not in _STOPWORDS and len(w) >= 3]

        uniq: List[str] = []
        seen = set()
        for kw in keywords:
            if kw in seen:
                continue
            seen.add(kw)
            uniq.append(kw)
        return uniq

    def predict(self, question: str, contexts: List[str]):
        keywords = self._extract_keywords(question)
        merged_context = " ".join([str(c).lower() for c in contexts])

        if not keywords:
            return (
                INSUFFICIENT,
                0.0,
                {
                    "근거": "키워드 추출 실패",
                    "키워드": [],
                    "히트수": 0,
                    "커버리지": 0.0,
                },
            )

        hit_keywords = [kw for kw in keywords if kw in merged_context]
        hit_count = len(hit_keywords)
        coverage_ratio = hit_count / max(1, len(keywords))

        is_sufficient = hit_count >= self.min_keyword_hits and coverage_ratio >= self.min_coverage_ratio
        label = SUFFICIENT if is_sufficient else INSUFFICIENT

        meta: Dict = {
            "키워드": keywords,
            "히트_키워드": hit_keywords,
            "히트수": hit_count,
            "커버리지": coverage_ratio,
            "기준_히트수": self.min_keyword_hits,
            "기준_커버리지": self.min_coverage_ratio,
        }
        return label, float(coverage_ratio), meta
