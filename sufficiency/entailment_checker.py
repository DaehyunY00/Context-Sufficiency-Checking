from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sufficiency.base import BaseChecker, INSUFFICIENT, SUFFICIENT


class EntailmentChecker(BaseChecker):
    """NLI 기반 문맥 충분성 판정기(MPS/CUDA/CPU)."""

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
        max_length: int = 512,
    ) -> None:
        requested_model = str(model_name).strip()
        self.model_name = requested_model
        self.threshold = float(sufficient_if_entail_prob_ge)
        self.max_length = int(max(32, max_length))
        self.device = self._resolve_device(device_preference or ["mps", "cpu"])
        self.tokenizer = None
        self.model = None
        self.entail_label_ids: List[int] = []
        self._load_model_with_fallback(requested_model=requested_model)

    @staticmethod
    def _resolve_device(device_preference: Sequence[str]) -> torch.device:
        for cand in [str(x).lower().strip() for x in device_preference]:
            if cand == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            if cand == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if cand == "cpu":
                return torch.device("cpu")
        return torch.device("cpu")

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

    @staticmethod
    def _infer_entailment_label_ids(model) -> List[int]:
        cfg = getattr(model, "config", None)
        label2id = getattr(cfg, "label2id", {}) or {}
        id2label = getattr(cfg, "id2label", {}) or {}

        ids: List[int] = []
        for label, idx in label2id.items():
            if "entail" in str(label).lower():
                try:
                    ids.append(int(idx))
                except (TypeError, ValueError):
                    pass
        if ids:
            return sorted(set(ids))

        for idx, label in id2label.items():
            if "entail" in str(label).lower():
                try:
                    ids.append(int(idx))
                except (TypeError, ValueError):
                    pass
        if ids:
            return sorted(set(ids))

        num_labels = int(getattr(cfg, "num_labels", 3) or 3)
        if num_labels >= 3:
            return [2]  # MNLI 관례: entailment=2
        if num_labels == 2:
            return [1]
        return [max(0, num_labels - 1)]

    def _load_model_with_fallback(self, requested_model: str) -> None:
        tried: List[str] = []
        errors: List[str] = []
        for cand in self._candidate_models(requested_model):
            tried.append(cand)
            try:
                tokenizer = AutoTokenizer.from_pretrained(cand)
                model = AutoModelForSequenceClassification.from_pretrained(cand)
                model.to(self.device)
                model.eval()
                entail_ids = self._infer_entailment_label_ids(model)
                self.model_name = cand
                self.tokenizer = tokenizer
                self.model = model
                self.entail_label_ids = entail_ids
                if cand != requested_model:
                    print(f"[NLI] 요청 모델 '{requested_model}' 대신 '{cand}'를 사용합니다.")
                print(f"[NLI] 장치={self.device}, entail_label_ids={self.entail_label_ids}")
                return
            except Exception as exc:  # pragma: no cover
                errors.append(f"{cand}: {exc}")

        err_preview = "\n".join(errors[:3])
        raise RuntimeError(
            "NLI 모델 로딩에 실패했습니다.\n"
            f"- 요청 모델: {requested_model}\n"
            f"- 시도 모델: {tried}\n"
            f"- 오류 예시:\n{err_preview}\n"
            "해결: 모델명/토큰/HF 인증 상태를 확인하세요."
        )

    def score_entailment(self, premise: str, hypothesis: str) -> float:
        if self.tokenizer is None or self.model is None:
            return 0.0

        inputs = self.tokenizer(
            str(premise),
            str(hypothesis),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]

        entail_prob = 0.0
        for idx in self.entail_label_ids:
            if 0 <= int(idx) < int(probs.shape[-1]):
                entail_prob = max(entail_prob, float(probs[int(idx)].item()))
        return float(entail_prob)

    def predict(self, question: str, contexts: List[str]):
        premise = " ".join([str(c) for c in contexts])[:2000]
        hypothesis = f"The context is sufficient to answer the question: {question}"
        entail_prob = self.score_entailment(premise=premise, hypothesis=hypothesis)
        label = SUFFICIENT if entail_prob >= self.threshold else INSUFFICIENT
        return label, entail_prob, {"entail_prob": entail_prob, "임계값": self.threshold}

