from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from generator.utils import format_contexts
from sufficiency.base import BaseChecker, INSUFFICIENT, SUFFICIENT


class LLMAutoraterChecker(BaseChecker):
    """LLM 프롬프트 기반 문맥 충분성 판정기."""

    name = "autorater"

    def __init__(
        self,
        text_generator,
        prompt_template_path: str | Path,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        do_sample: bool = False,
        parse_fail_policy: str = "insufficient",
        confidence_threshold: float = 0.0,
    ) -> None:
        self.text_generator = text_generator
        self.prompt_template_path = Path(prompt_template_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)
        self.parse_fail_policy = str(parse_fail_policy).lower().strip()
        self.confidence_threshold = float(confidence_threshold)
        self.template = self._load_template(self.prompt_template_path)

    @staticmethod
    def _load_template(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"autorater 템플릿 파일이 없습니다: {path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _extract_json(raw_text: str) -> Optional[Dict[str, Any]]:
        text = raw_text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 모델이 규칙을 어긴 경우 대비한 최소 복구 파싱
        match = re.search(r"\{.*\}", text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _render_prompt(self, question: str, contexts: List[str]) -> str:
        context_block = format_contexts(contexts, max_chars=2200)
        prompt = self.template.replace("{question}", question)
        prompt = prompt.replace("{context}", context_block)
        return prompt

    def _parse_fail_result(self, raw_output: str):
        label = INSUFFICIENT if self.parse_fail_policy == "insufficient" else INSUFFICIENT
        return (
            label,
            0.0,
            {
                "파싱오류": "JSON 파싱 실패",
                "원본출력": raw_output,
                "정책": self.parse_fail_policy,
            },
        )

    def predict(self, question: str, contexts: List[str]):
        prompt = self._render_prompt(question=question, contexts=contexts)
        raw_output = self.text_generator.generate_from_prompt(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
        )

        parsed = self._extract_json(raw_output)
        if parsed is None:
            return self._parse_fail_result(raw_output)

        label = str(parsed.get("label", INSUFFICIENT)).upper().strip()
        if label not in {SUFFICIENT, INSUFFICIENT}:
            label = INSUFFICIENT

        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        missing_info = parsed.get("missing_info", [])
        if not isinstance(missing_info, list):
            missing_info = []
        missing_info = [str(x)[:64] for x in missing_info]

        if label == SUFFICIENT and confidence < self.confidence_threshold:
            label = INSUFFICIENT

        if label == SUFFICIENT:
            missing_info = []

        return (
            label,
            confidence,
            {
                "missing_info": missing_info,
                "원본출력": raw_output,
                "confidence_threshold": self.confidence_threshold,
            },
        )
