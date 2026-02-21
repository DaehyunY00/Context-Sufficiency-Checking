from __future__ import annotations

import re
from collections import Counter
from typing import List

from sufficiency.base import BaseChecker, INSUFFICIENT, SUFFICIENT


def _normalize_text(text: str) -> str:
    normalized = str(text).lower().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class SelfConsistencyChecker(BaseChecker):
    """다중 샘플 생성의 불일치 정도로 충분성 판정."""

    name = "self_consistency"

    def __init__(
        self,
        generator,
        n_samples: int = 5,
        temperature: float = 0.7,
        disagreement_threshold: float = 0.6,
        abstain_text: str = "모르겠습니다.",
        prompt_style: str = "qa_short_ko",
    ) -> None:
        self.generator = generator
        self.n_samples = int(n_samples)
        self.temperature = float(temperature)
        self.disagreement_threshold = float(disagreement_threshold)
        self.abstain_text = abstain_text
        self.prompt_style = prompt_style

    def predict(self, question: str, contexts: List[str]):
        answers: List[str] = []
        for i in range(self.n_samples):
            out = self.generator.generate_answer(
                question=question,
                contexts=contexts,
                abstain_text=self.abstain_text,
                prompt_style=self.prompt_style,
                temperature=self.temperature,
                do_sample=True,
                seed=1000 + i,
            )
            answers.append(out)

        normalized = [_normalize_text(a) for a in answers]
        counter = Counter(normalized)
        _, max_count = counter.most_common(1)[0]

        majority_ratio = max_count / max(1, len(normalized))
        disagreement = 1.0 - majority_ratio
        abstain_ratio = sum(1 for a in answers if a.strip() == self.abstain_text) / max(1, len(answers))

        label = INSUFFICIENT if disagreement >= self.disagreement_threshold else SUFFICIENT
        if abstain_ratio >= 0.5:
            label = INSUFFICIENT

        return (
            label,
            float(disagreement),
            {
                "샘플답변": answers,
                "정규화답변": normalized,
                "불일치도": disagreement,
                "다수결비율": majority_ratio,
                "모르겠습니다_비율": abstain_ratio,
                "임계값": self.disagreement_threshold,
            },
        )
