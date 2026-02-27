from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import pipeline

from sufficiency.base import BaseChecker, INSUFFICIENT, SUFFICIENT


class EntailmentChecker(BaseChecker):
    """NLI 기반 문맥 충분성 판정기(옵션)."""

    name = "entailment"

    def __init__(
        self,
        model_name: str = "roberta-base-mnli",
        sufficient_if_entail_prob_ge: float = 0.6,
        device_preference: Sequence[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = float(sufficient_if_entail_prob_ge)
        self.device_index = self._resolve_pipeline_device(device_preference or ["mps", "cpu"])
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=self.device_index,
            return_all_scores=True,
            truncation=True,
        )

    @staticmethod
    def _resolve_pipeline_device(device_preference: Sequence[str]) -> int:
        # transformers pipeline은 MPS 인덱스를 직접 받지 않으므로 CPU(-1) 사용
        for cand in [str(x).lower().strip() for x in device_preference]:
            if cand == "cuda" and torch.cuda.is_available():
                return 0
        return -1

    def score_entailment(self, premise: str, hypothesis: str) -> float:
        result = self.classifier({"text": premise, "text_pair": hypothesis})[0]
        entail_prob = 0.0
        for item in result:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.0))
            if "entail" in label or label == "label_2":
                entail_prob = max(entail_prob, score)
        return float(entail_prob)

    def predict(self, question: str, contexts: List[str]):
        premise = " ".join([str(c) for c in contexts])[:2000]
        hypothesis = f"The context is sufficient to answer the question: {question}"
        entail_prob = self.score_entailment(premise=premise, hypothesis=hypothesis)

        label = SUFFICIENT if entail_prob >= self.threshold else INSUFFICIENT
        return label, entail_prob, {"entail_prob": entail_prob, "임계값": self.threshold}
