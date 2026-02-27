from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
        max_parse_retries: int = 1,
        max_context_chars: int = 2200,
    ) -> None:
        self.text_generator = text_generator
        self.prompt_template_path = Path(prompt_template_path)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)
        self.parse_fail_policy = str(parse_fail_policy).lower().strip()
        self.confidence_threshold = float(confidence_threshold)
        self.max_parse_retries = max(0, int(max_parse_retries))
        self.max_context_chars = max(400, int(max_context_chars))
        self.template = self._load_template(self.prompt_template_path)

    @staticmethod
    def _load_template(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"autorater 템플릿 파일이 없습니다: {path}")
        return path.read_text(encoding="utf-8")

    @classmethod
    def _try_parse_dict(cls, candidate: str) -> Optional[Dict[str, Any]]:
        text = str(candidate).strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

        return None

    @classmethod
    def _extract_json(cls, raw_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        text = str(raw_text).strip()
        if not text:
            return None, "empty_output"

        # Strict mode: JSON 한 줄 전체가 스키마 객체여야 하며, 부가 텍스트를 허용하지 않는다.
        if "\n" in text or "\r" in text:
            return None, "strict_json_line_violation"
        if not (text.startswith("{") and text.endswith("}")):
            return None, "strict_json_boundary_violation"

        parsed = cls._try_parse_dict(text)
        if parsed is None:
            return None, "strict_json_decode_failed"
        return parsed, "strict_json"

    def _render_prompt(self, question: str, contexts: List[str]) -> str:
        context_block = format_contexts(contexts, max_chars=self.max_context_chars)
        prompt = self.template.replace("{question}", question)
        prompt = prompt.replace("{context}", context_block)
        return prompt

    def _render_retry_prompt(self, question: str, contexts: List[str]) -> str:
        context_block = format_contexts(contexts, max_chars=self.max_context_chars)
        return (
            "이전 출력이 JSON 스키마를 지키지 않았습니다. 아래 규칙을 다시 따르세요.\n"
            "출력은 JSON 한 줄만 허용합니다.\n"
            "스키마: {\"label\":\"SUFFICIENT|INSUFFICIENT\",\"confidence\":0.0,\"missing_info\":[\"...\"]}\n"
            f"질문: {question}\n"
            f"문맥:\n{context_block}\n"
            "출력:"
        )

    def _parse_fail_result(
        self,
        raw_output: str,
        reason: str = "JSON 파싱 실패",
        raw_outputs: Optional[List[str]] = None,
        parse_method: str = "json_parse_failed",
    ):
        label = INSUFFICIENT if self.parse_fail_policy == "insufficient" else INSUFFICIENT
        history = raw_outputs or [raw_output]
        return (
            label,
            0.0,
            {
                "파싱오류": reason,
                "원본출력": raw_output,
                "원본출력_시도별": history,
                "정책": self.parse_fail_policy,
                "파싱방식": parse_method,
                "재시도횟수": max(0, len(history) - 1),
            },
        )

    def predict(self, question: str, contexts: List[str]):
        raw_outputs: List[str] = []
        parsed: Optional[Dict[str, Any]] = None
        parse_method = "not_started"

        for attempt in range(self.max_parse_retries + 1):
            prompt = (
                self._render_prompt(question=question, contexts=contexts)
                if attempt == 0
                else self._render_retry_prompt(question=question, contexts=contexts)
            )
            raw_output = self.text_generator.generate_from_prompt(
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
            )
            raw_outputs.append(raw_output)
            parsed, parse_method = self._extract_json(raw_output)
            if parsed is not None:
                break

        if parsed is None:
            return self._parse_fail_result(
                raw_output=raw_outputs[-1] if raw_outputs else "",
                raw_outputs=raw_outputs,
                parse_method=parse_method,
            )

        if not isinstance(parsed, dict):
            return self._parse_fail_result(
                raw_output=raw_outputs[-1] if raw_outputs else "",
                reason="JSON 객체 스키마가 아님",
                raw_outputs=raw_outputs,
                parse_method=parse_method,
            )

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
                "원본출력": raw_outputs[-1] if raw_outputs else "",
                "원본출력_시도별": raw_outputs,
                "confidence_threshold": self.confidence_threshold,
                "파싱방식": parse_method,
                "재시도횟수": max(0, len(raw_outputs) - 1),
            },
        )
