from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import pipeline

from sufficiency.base import BaseChecker, INSUFFICIENT, SUFFICIENT


class EntailmentChecker(BaseChecker):
    """NLI 기반 문맥 충분성 판정기(옵션)."""

    name = "entailment"
    LEGACY_ALIAS = {
        "roberta-base-mnli": "FacebookAI/roberta-large-mnli",
    }
    FALLBACK_MODELS = [
        "cross-encoder/nli-distilroberta-base",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "FacebookAI/roberta-large-mnli",
    ]

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-distilroberta-base",
        sufficient_if_entail_prob_ge: float = 0.6,
        device_preference: Sequence[str] | None = None,
    ) -> None:
        requested_model = str(model_name).strip()
        self.model_name = requested_model
        self.threshold = float(sufficient_if_entail_prob_ge)
        self.device_index = self._resolve_pipeline_device(device_preference or ["mps", "cpu"])
        self.classifier = self._build_classifier_with_fallback(requested_model=requested_model)

    def _candidate_models(self, requested_model: str) -> List[str]:
        primary = self.LEGACY_ALIAS.get(requested_model, requested_model)
        ordered = [primary] + list(self.FALLBACK_MODELS)
        uniq: List[str] = []
        seen = set()
        for m in ordered:
            key = str(m).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            uniq.append(key)
        return uniq

    def _build_classifier_with_fallback(self, requested_model: str):
        tried: List[str] = []
        errors: List[str] = []

        for cand in self._candidate_models(requested_model):
            tried.append(cand)
            try:
                clf = pipeline(
                    "text-classification",
                    model=cand,
                    device=self.device_index,
                    top_k=None,
                    truncation=True,
                    token=False,  # 공개 모델 접근 시 로컬 잘못된 토큰을 강제 사용하지 않음
                )
                self.model_name = cand
                if cand != requested_model:
                    print(f"[NLI] 요청 모델 '{requested_model}' 대신 '{cand}'를 사용합니다.")
                return clf
            except Exception as exc:  # pragma: no cover
                errors.append(f"{cand}: {exc}")

        err_preview = "\n".join(errors[:3])
        raise RuntimeError(
            "NLI 모델 로딩에 실패했습니다.\n"
            f"- 요청 모델: {requested_model}\n"
            f"- 시도 모델: {tried}\n"
            f"- 오류 예시:\n{err_preview}\n"
            "해결: 모델명을 공개 모델로 지정하거나 HF 인증정보를 확인하세요."
        )

    @staticmethod
    def _resolve_pipeline_device(device_preference: Sequence[str]) -> int:
        # transformers pipeline은 MPS 인덱스를 직접 받지 않으므로 CPU(-1) 사용
        for cand in [str(x).lower().strip() for x in device_preference]:
            if cand == "cuda" and torch.cuda.is_available():
                return 0
        return -1

    @staticmethod
    def _normalize_classifier_output(raw_output) -> List[dict]:
        """
        transformers 버전에 따라 text-classification 출력 형태가 달라질 수 있어
        가능한 구조를 모두 정규화한다.
        - list[list[dict]]
        - list[dict]
        - dict
        """
        if isinstance(raw_output, list):
            if not raw_output:
                return []
            first = raw_output[0]
            if isinstance(first, list):
                return [x for x in first if isinstance(x, dict)]
            if isinstance(first, dict):
                return [x for x in raw_output if isinstance(x, dict)]
            return []

        if isinstance(raw_output, dict):
            return [raw_output]

        return []

    def score_entailment(self, premise: str, hypothesis: str) -> float:
        raw = self.classifier({"text": premise, "text_pair": hypothesis})
        result = self._normalize_classifier_output(raw)
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
