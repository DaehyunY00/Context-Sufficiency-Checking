from __future__ import annotations

import hashlib
import re
from typing import Dict, Iterable, List, Tuple


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text)).strip()


def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    """문자 단위 슬라이딩 윈도우 청크."""
    clean = normalize_whitespace(text)
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks: List[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(clean), step):
        end = min(len(clean), start + chunk_size)
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
    return chunks


def _extract_title_sentences(context_obj) -> List[Tuple[str, List[str]]]:
    """HotpotQA의 context 포맷(dict/list) 모두 지원."""
    pairs: List[Tuple[str, List[str]]] = []

    if isinstance(context_obj, dict):
        titles = context_obj.get("title", [])
        sentences = context_obj.get("sentences", [])
        for title, sent_list in zip(titles, sentences):
            sent_items = sent_list if isinstance(sent_list, list) else [str(sent_list)]
            pairs.append((str(title), [str(s) for s in sent_items]))
        return pairs

    if isinstance(context_obj, list):
        for item in context_obj:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0])
                sent_list = item[1]
                sent_items = sent_list if isinstance(sent_list, list) else [str(sent_list)]
                pairs.append((title, [str(s) for s in sent_items]))
    return pairs


def extract_hotpot_context_paragraphs(example: Dict) -> List[Dict]:
    """HotpotQA 샘플에서 title/text 문단 리스트를 추출."""
    context_obj = example.get("context", {})
    title_sents = _extract_title_sentences(context_obj)
    paragraphs: List[Dict] = []

    for title, sents in title_sents:
        paragraph = normalize_whitespace(" ".join(sents))
        if paragraph:
            paragraphs.append({"title": title, "text": paragraph})
    return paragraphs


def build_corpus_from_examples(
    examples: Iterable[Dict],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> List[Dict]:
    """질문 샘플의 context를 병합해 검색 코퍼스 구성."""
    seen = set()
    corpus: List[Dict] = []

    for ex in examples:
        qid = str(ex.get("question_id", "unknown"))
        for para in ex.get("contexts", []):
            title = str(para.get("title", ""))
            text = str(para.get("text", ""))
            for idx, piece in enumerate(chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)):
                digest = hashlib.md5(f"{title}::{piece}".encode("utf-8")).hexdigest()[:12]
                if digest in seen:
                    continue
                seen.add(digest)
                corpus.append(
                    {
                        "doc_id": f"{title}::{qid}::{idx}::{digest}",
                        "title": title,
                        "text": piece,
                    }
                )
    return corpus
