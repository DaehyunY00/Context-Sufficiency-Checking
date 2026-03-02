from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Set

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

    def __init__(
        self,
        min_keyword_hits: int = 2,
        min_coverage_ratio: float = 0.5,
        variant: str = "h1",
    ) -> None:
        self.min_keyword_hits = int(min_keyword_hits)
        self.min_coverage_ratio = float(min_coverage_ratio)
        self.variant = self._normalize_variant(variant)

    @staticmethod
    def _normalize_variant(variant: str) -> str:
        alias = {
            "h1": "h1",
            "binary": "h1",
            "keyword": "h1",
            "h2": "h2",
            "tfidf": "h2",
            "tf_idf": "h2",
            "h3": "h3",
            "entity": "h3",
            "ner": "h3",
        }
        return alias.get(str(variant).strip().lower(), "h1")

    def _tokenize(self, text: str) -> List[str]:
        candidates = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", str(text).lower())
        return [w for w in candidates if w not in _STOPWORDS and len(w) >= 3]

    def _extract_keywords(self, question: str) -> List[str]:
        tokens = self._tokenize(question)
        return self._unique(tokens)

    @staticmethod
    def _unique(tokens: Sequence[str]) -> List[str]:
        uniq: List[str] = []
        seen = set()
        for kw in tokens:
            if kw in seen:
                continue
            seen.add(kw)
            uniq.append(kw)
        return uniq

    def _extract_entities(self, text: str) -> List[str]:
        raw = str(text)
        phrase_matches = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", raw)
        acronym_matches = re.findall(r"\b[A-Z]{2,}(?:\d+)?\b", raw)

        candidates = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", raw.lower())
        fallback_keywords = [w for w in candidates if w not in _STOPWORDS and len(w) >= 3]

        entities: List[str] = []
        for item in phrase_matches + acronym_matches:
            norm = re.sub(r"\s+", " ", str(item).strip()).lower()
            if not norm:
                continue
            if norm in _STOPWORDS:
                continue
            entities.append(norm)

        # 질문에 고유명사 패턴이 거의 없을 때 완전 붕괴를 막기 위한 약한 폴백
        if not entities:
            entities = fallback_keywords[:4]

        return self._unique(entities)

    def _predict_h1(self, question: str, contexts: List[str]):
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
                    "변형": "h1",
                },
            )

        hit_keywords = [kw for kw in keywords if kw in merged_context]
        hit_count = len(hit_keywords)
        coverage_ratio = hit_count / max(1, len(keywords))

        is_sufficient = hit_count >= self.min_keyword_hits and coverage_ratio >= self.min_coverage_ratio
        label = SUFFICIENT if is_sufficient else INSUFFICIENT

        meta: Dict = {
            "변형": "h1",
            "키워드": keywords,
            "히트_키워드": hit_keywords,
            "히트수": hit_count,
            "커버리지": coverage_ratio,
            "기준_히트수": self.min_keyword_hits,
            "기준_커버리지": self.min_coverage_ratio,
        }
        return label, float(coverage_ratio), meta

    def _predict_h2(self, question: str, contexts: List[str]):
        q_tokens = self._tokenize(question)
        if not q_tokens:
            return (
                INSUFFICIENT,
                0.0,
                {
                    "근거": "질문 토큰 추출 실패",
                    "변형": "h2",
                    "가중커버리지": 0.0,
                },
            )

        q_tf = Counter(q_tokens)
        q_terms = list(q_tf.keys())

        ctx_doc_sets = [set(self._tokenize(ctx)) for ctx in contexts if str(ctx).strip()]
        if not ctx_doc_sets:
            return (
                INSUFFICIENT,
                0.0,
                {
                    "근거": "문맥 토큰 추출 실패",
                    "변형": "h2",
                    "가중커버리지": 0.0,
                },
            )

        n_docs = len(ctx_doc_sets)
        hit_terms = []
        weighted_num = 0.0
        weighted_den = 0.0
        idf_meta: Dict[str, float] = {}

        for t in q_terms:
            df = sum(1 for d in ctx_doc_sets if t in d)
            idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
            tfidf = float(q_tf[t]) * idf
            weighted_den += tfidf
            idf_meta[t] = idf
            if df > 0:
                weighted_num += tfidf
                hit_terms.append(t)

        weighted_coverage = weighted_num / max(weighted_den, 1e-12)
        hit_count = len(hit_terms)

        is_sufficient = hit_count >= 1 and weighted_coverage >= self.min_coverage_ratio
        label = SUFFICIENT if is_sufficient else INSUFFICIENT

        meta: Dict = {
            "변형": "h2",
            "질문토큰수": len(q_tokens),
            "질문유니크토큰수": len(q_terms),
            "히트_토큰": hit_terms,
            "히트수": hit_count,
            "idf": idf_meta,
            "가중커버리지": weighted_coverage,
            "기준_커버리지": self.min_coverage_ratio,
        }
        return label, float(weighted_coverage), meta

    def _predict_h3(self, question: str, contexts: List[str]):
        q_entities = self._extract_entities(question)
        ctx_entities = self._extract_entities(" ".join([str(c) for c in contexts]))

        if not q_entities:
            return (
                INSUFFICIENT,
                0.0,
                {
                    "근거": "질문 개체 추출 실패",
                    "질문개체": [],
                    "문맥개체": ctx_entities,
                    "변형": "h3",
                },
            )

        q_set = set(q_entities)
        c_set = set(ctx_entities)
        hit_entities = sorted(q_set & c_set)
        entity_ratio = len(hit_entities) / max(len(q_set), 1)

        is_sufficient = entity_ratio >= self.min_coverage_ratio
        label = SUFFICIENT if is_sufficient else INSUFFICIENT

        meta: Dict = {
            "변형": "h3",
            "질문개체": sorted(q_set),
            "문맥개체": sorted(c_set),
            "히트개체": hit_entities,
            "개체매칭비율": entity_ratio,
            "기준_커버리지": self.min_coverage_ratio,
        }
        return label, float(entity_ratio), meta

    def predict(self, question: str, contexts: List[str]):
        if self.variant == "h2":
            return self._predict_h2(question, contexts)
        if self.variant == "h3":
            return self._predict_h3(question, contexts)
        return self._predict_h1(question, contexts)
