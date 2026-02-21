from __future__ import annotations

import re
from typing import Iterable, List


def format_contexts(contexts: Iterable[str], max_chars: int = 1800) -> str:
    lines: List[str] = []
    total = 0
    for i, ctx in enumerate(contexts, start=1):
        clean = re.sub(r"\s+", " ", str(ctx)).strip()
        if not clean:
            continue
        line = f"[{i}] {clean}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)


def build_qa_prompt(question: str, contexts: Iterable[str], abstain_text: str = "모르겠습니다.") -> str:
    context_block = format_contexts(contexts)
    return (
        "다음 문맥만 근거로 질문에 짧게 답하세요.\n"
        f"문맥이 불충분하면 정확히 '{abstain_text}'라고 답하세요.\n\n"
        f"질문: {question}\n\n"
        f"문맥:\n{context_block}\n\n"
        "답변:"
    )


def clean_generated_text(text: str) -> str:
    clean = re.sub(r"\s+", " ", str(text)).strip()
    return clean.strip('"')
