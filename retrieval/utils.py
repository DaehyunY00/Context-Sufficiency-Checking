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


def _extract_generic_context_paragraphs(context_obj, default_title: str = "문서") -> List[Dict]:
    paragraphs: List[Dict] = []
    if isinstance(context_obj, str):
        text = normalize_whitespace(context_obj)
        if text:
            paragraphs.append({"title": default_title, "text": text})
        return paragraphs

    if isinstance(context_obj, dict):
        for key in ["text", "paragraph", "passage", "content"]:
            if key in context_obj:
                value = context_obj.get(key)
                if isinstance(value, list):
                    for item in value:
                        text = normalize_whitespace(item)
                        if text:
                            paragraphs.append({"title": str(context_obj.get("title", default_title)), "text": text})
                else:
                    text = normalize_whitespace(value)
                    if text:
                        paragraphs.append({"title": str(context_obj.get("title", default_title)), "text": text})
                break
        return paragraphs

    if isinstance(context_obj, list):
        for idx, item in enumerate(context_obj):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0]) or f"{default_title}_{idx}"
                second = item[1]
                if isinstance(second, list):
                    text = normalize_whitespace(" ".join([str(x) for x in second]))
                else:
                    text = normalize_whitespace(second)
                if text:
                    paragraphs.append({"title": title, "text": text})
            elif isinstance(item, dict):
                title = str(item.get("title", item.get("source", default_title)))
                text = normalize_whitespace(
                    item.get("text", item.get("paragraph", item.get("passage", item.get("content", ""))))
                )
                if text:
                    paragraphs.append({"title": title, "text": text})
            else:
                text = normalize_whitespace(item)
                if text:
                    paragraphs.append({"title": f"{default_title}_{idx}", "text": text})
    return paragraphs


def extract_2wiki_context_paragraphs(example: Dict) -> List[Dict]:
    """2WikiMultiHopQA 계열 포맷에서 문맥 문단을 추출한다."""
    for key in ["context", "contexts", "paragraphs", "passages", "documents", "wiki_context"]:
        if key not in example:
            continue
        paras = _extract_generic_context_paragraphs(example.get(key), default_title="2wiki")
        if paras:
            return paras
    return []


def extract_nq_context_paragraphs(example: Dict) -> List[Dict]:
    document = example.get("document", {})
    if not isinstance(document, dict):
        return []

    title = str(document.get("title", "natural_questions")).strip() or "natural_questions"
    tokens_obj = document.get("tokens", {})
    tokens: List[str] = []
    html_flags: List[bool] = []

    if isinstance(tokens_obj, dict):
        raw_tokens = tokens_obj.get("token", tokens_obj.get("tokens", []))
        raw_html = tokens_obj.get("is_html", [])
        if isinstance(raw_tokens, list):
            tokens = [str(t) for t in raw_tokens]
        if isinstance(raw_html, list):
            html_flags = [bool(x) for x in raw_html]
    elif isinstance(tokens_obj, list):
        for item in tokens_obj:
            if isinstance(item, dict):
                tokens.append(str(item.get("token", item.get("text", ""))))
                html_flags.append(bool(item.get("is_html", False)))
            else:
                tokens.append(str(item))
                html_flags.append(False)
    else:
        raw_text = normalize_whitespace(document.get("text", document.get("html", "")))
        if raw_text:
            return [{"title": title, "text": raw_text}]

    if tokens and len(html_flags) == len(tokens):
        tokens = [tok for tok, is_html in zip(tokens, html_flags) if not is_html]
    tokens = [str(t) for t in tokens if str(t).strip()]
    if not tokens:
        return []

    # Natural Questions 문서는 길기 때문에 120토큰 단위로 문단화한다.
    paragraphs: List[Dict] = []
    step = 120
    for i in range(0, len(tokens), step):
        block = normalize_whitespace(" ".join(tokens[i : i + step]))
        if block:
            paragraphs.append({"title": f"{title}_{i//step}", "text": block})
    return paragraphs


def extract_context_paragraphs(example: Dict, dataset_name: str) -> List[Dict]:
    name = str(dataset_name).lower().strip()
    if name in {"hotpotqa", "hotpot_qa"}:
        return extract_hotpot_context_paragraphs(example)
    if name in {"2wikimultihopqa", "2wiki", "2wiki_multihop_qa"}:
        return extract_2wiki_context_paragraphs(example)
    if name in {"natural_questions", "nq", "naturalquestions"}:
        return extract_nq_context_paragraphs(example)

    for key in ["context", "contexts", "documents", "paragraphs", "passages"]:
        if key in example:
            paras = _extract_generic_context_paragraphs(example.get(key), default_title=name or "문서")
            if paras:
                return paras
    return []


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
